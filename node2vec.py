from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd

import dataset_old as dataset
import custom


def write(filename, index, data):
    df = pd.DataFrame(data)
    df["idx"] = index
    df = df.set_index("idx")
    df.to_csv(filename, header=False, sep="\t", encoding="utf-8")


class JointModel(object):
    def __init__(self, data_dir):
        # IO files
        self.template_file = data_dir + "/templates.txt"
        self.word_file = data_dir + "/words.txt"
        self.log = None

        # hyper parameters
        self.max_epoch = 1
        self.batch_size = 64
        self.learning_rate = 0.1
        self.checkpoint = 1
        self.template_dim = 5
        self.word_dim = 10
        self.neg_samples = 16  # Number of negative samples
        self.optimizer_class = tf.train.GradientDescentOptimizer
        self.tied_weights = False
        # how to combine training losses on the tree embedding and word embedding
        # "sum" for sum of losses
        # "alternate" for alternating between losses for each training step
        # default for customized NCE loss directly on the combined embedding (no component losses)
        self.loss_type = None

        # must be specified before building
        self.vocab_size = None
        self.num_templates = None

    def embed(self, template, word):
        return tf.concat([tf.nn.embedding_lookup(self.template_embedding, template),
                          tf.nn.embedding_lookup(self.word_embedding, word)],
                         axis=1)

    def optimize(self, loss):
        return self.optimizer_class(learning_rate=self.learning_rate).minimize(loss)

    def build(self, data=None):
        if data is not None:
            self.vocab_size = data.input_count()
            self.num_templates = data.tag_count()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_template = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.input_word = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.output_template = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.output_word = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            self.template_embedding = tf.Variable(tf.random_uniform([self.num_templates, self.template_dim], -1.0, 1.0))
            self.word_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.word_dim], -1.0, 1.0))

            template_biases = tf.Variable(tf.zeros([self.num_templates]))
            word_biases = tf.Variable(tf.zeros([self.vocab_size]))

            if self.loss_type in ["sum", "alternate"]:
                self.input_embedding = self.embed(self.input_template, self.input_word)
                embedding_dim = self.input_embedding.get_shape().as_list()[-1]
                template_weights = tf.Variable(tf.random_uniform([self.num_templates, embedding_dim], -1.0, 1.0))
                self.template_loss = tf.reduce_mean(tf.nn.nce_loss(template_weights, template_biases,
                                                                   self.output_template, self.input_embedding,
                                                                   self.neg_samples, self.num_templates))
                word_weights = tf.Variable(tf.random_uniform([self.vocab_size, embedding_dim], -1.0, 1.0))
                self.word_loss = tf.reduce_mean(tf.nn.nce_loss(word_weights, word_biases,
                                                               self.output_word, self.input_embedding,
                                                               self.neg_samples, self.vocab_size))
            elif self.loss_type == "cartesian":
                self.input_embedding = self.embed(self.input_template, self.input_word)
                embedding_dim = self.input_embedding.get_shape().as_list()[-1]
                nce_weights = tf.Variable(tf.random_uniform([self.num_templates * self.vocab_size, embedding_dim], -1.0, 1.0))
                nce_biases = tf.Variable(tf.zeros([self.num_templates * self.vocab_size]))
                output_embedding = self.output_template * self.vocab_size + self.output_word # WIP
                self.loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases,
                                                          output_embedding, 
                                                          self.input_embedding,
                                                          self.neg_samples, self.num_templates * self.vocab_size))
            elif self.loss_type == "independent":
                self.input_embedding = self.embed(self.input_template, self.input_word)
                out_template_embedding = tf.Variable(tf.random_uniform([self.num_templates, self.template_dim], -1.0, 1.0))
                out_word_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.word_dim], -1.0, 1.0))
                def out_embed(template, word):
                    return tf.concat([tf.nn.embedding_lookup(out_template_embedding, template),
                                      tf.nn.embedding_lookup(out_word_embedding, word)],
                                     axis=1)
                self.loss = tf.reduce_mean(custom.nce_loss(out_embed,
                                                           [template_biases, word_biases],
                                                           [self.output_template, self.output_word],
                                                           self.input_embedding,
                                                           self.neg_samples,
                                                           [self.num_templates, self.vocab_size]))
            else:
                self.input_embedding = self.embed(self.input_template, self.input_word)
                self.loss = tf.reduce_mean(custom.nce_loss(self.embed,
                                                           [template_biases, word_biases],
                                                           [self.output_template, self.output_word],
                                                           self.input_embedding,
                                                           self.neg_samples,
                                                           [self.num_templates, self.vocab_size]))
                

    def new_session(self):
        return tf.Session(graph=self.graph)

    def train(self, session, data, max_epoch=None):
        if max_epoch is not None:
            self.max_epoch = max_epoch

        if self.loss_type == "sum":
            self.loss = self.template_loss + self.word_loss
            optimizers = [self.optimize(self.loss)]
        if self.loss_type == "alternate":
            self.loss = self.word_loss
            optimizers = [self.optimize(self.template_loss), self.optimize(self.word_loss)]
        else:
            optimizers = [self.optimize(self.loss)]
        
        average_loss = 0
        for epoch in range(max_epoch):
            final_loss = []
            step = 0
            while True:
                step += 1
                parent_template, parent_word, child_template, child_word, action = data.gen_batch(self.batch_size)
                if parent_template is None:
                    # print("\r", "Epoch ", epoch, " final loss ", np.mean(final_loss))
                    data.reset()
                    break

                feed_dict = {self.input_template : parent_template,
                             self.input_word: parent_word,
                             self.output_template: np.expand_dims(child_template, 1),
                             self.output_word: np.expand_dims(child_word, 1)}

                session.run(optimizers, feed_dict=feed_dict)

                average_loss += session.run(self.loss, feed_dict=feed_dict)
                if step % self.checkpoint == 0:
                    average_loss /= self.checkpoint
                    final_loss.append(average_loss)
                    print("\r", "Epoch ", epoch, " Loss ", average_loss, end="")
                    if self.log is not None:
                        self.log.write("Epoch " + str(epoch) + " Loss " + str(average_loss) + "\n")
                        self.log.flush
                    average_loss = 0

        return np.mean(final_loss)

    def write_embeddings(self, session, data):
        write(self.template_file, data.tag_vocab(), session.run(self.template_embedding))
        write(self.word_file, data.input_vocab(), session.run(self.word_embedding))

        
def train_file(filename, learning_rate):
    for loss_type in [None]:
        trainer = JointModel(data_dir="data")
        trainer.batch_size = 128
        trainer.checkpoint = 500
        trainer.learning_rate = learning_rate
        trainer.loss_type = loss_type
        trainer.optimizer_class = tf.train.GradientDescentOptimizer
    
        trainer.word_dim = 50
        trainer.template_dim = 10
    
        text = dataset.TextWindowDataset("data/penn.text", min_count=10)
    
        data = treeset.Edges()
        data.words.index_map = text.dictionary
        data.words.label_map = text.reverse_dictionary
        data.words.import_index = True
        data.read(filename)
        
        print(data.num_templates(), data.vocab_size())
    
        trainer.build(data)
    
        with trainer.new_session() as session, open("data/train_log." + str(loss_type) , "w") as log_file:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            old_loss = None
            for epoch in range(0, 170):
                timer = time.time()
                loss = trainer.train(session, data, max_epoch=1)
                print("")
                if old_loss is not None and loss >= old_loss:
                  trainer.learning_rate = trainer.learning_rate * 0.98
                  print("New learning rate=", trainer.learning_rate)
                old_loss = loss
                log_file.write(str(time.time() - timer) + " : " + str(loss) + "\n")
                log_file.flush()
                
                saver.save(session, "data/params.tf")
                trainer.word_file = "data/words." + str(loss_type) + "." + str(epoch)
                trainer.write_embeddings(session, data)


def train_new():
    trainer = JointModel(data_dir="data")
    trainer.batch_size = 5000
    trainer.checkpoint = 200
    trainer.learning_rate = 1.0
    trainer.loss_type = None
    trainer.optimizer_class = tf.train.GradientDescentOptimizer
    trainer.word_dim = 50
    trainer.template_dim = 10
    trainer.neg_samples = 8
    lr_decay = 0.9999

    train_dir = "/home/vu/bilstm_stagging/result"
    filenames = sorted([os.path.join(train_dir, name) for name in os.listdir(train_dir) if name.endswith(".txt")])
    data = dataset.TaggedTextFilesDataset(filenames, min_count=10)
    data.num_skips = 10
    data.skip_window = 5
    print("Vocab size", data.input_count(), data.tag_count())
    trainer.build(data)

    with trainer.new_session() as session, open("data/train_log.txt", "w") as log_file:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        old_loss = None
        for epoch in range(0, 5):
            for i, filename in enumerate(filenames):
                timer = time.time()
                data.load_file(filename)
                loss = trainer.train(session, data, max_epoch=1)
                print("\r", "Epoch", epoch, i, "/", len(filenames), filename, time.time() - timer)
                print("")
                if old_loss is not None and loss >= old_loss:
                    trainer.learning_rate = trainer.learning_rate * lr_decay
                    print("New learning rate=", trainer.learning_rate)
                old_loss = loss
                log_file.write(str(epoch) + ":" + filename + ":" + str(time.time() - timer) + " : " + str(loss) + "\n")
                log_file.flush()
                saver.save(session, "data/params.tf")
            trainer.word_file = "data/words." + str(epoch)
            trainer.write_embeddings(session, data)


if __name__ == "__main__":
    train_new()

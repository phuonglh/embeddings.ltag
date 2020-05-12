from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd

import dataset
import custom_2

def write(filename, index, data):
    df = pd.DataFrame(data)
    df["idx"] = index
    df = df.set_index("idx")
    df.to_csv(filename, header=False, sep="\t")


class Word2Vec(object):
    def __init__(self, data_dir):
        # IO files
        self.invec_file = data_dir + "/invec.txt"
        self.outvec_file = data_dir + "/outvec.txt"

        # hyper parameters
        self.max_epoch = 1
        self.batch_size = 64
        self.learning_rate = 0.1
        self.checkpoint = 1
        self.max_steps = None
        self.embedding_dim = 50  # Dimension of the embedding vector.
        self.neg_samples = 16  # Number of negative samples
        self.optimizer_class = tf.train.GradientDescentOptimizer

        # must be specified before building
        self.input_count = None
        self.output_count = None
        self.embed_input = None
        self.embed_output = None

    def build(self, data=None):
        if data is not None:
            self.input_count = data.input_count()
            self.output_count = data.output_count()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Construct the variables.
            self.input_embedding = tf.Variable(tf.random_uniform([self.input_count, self.embedding_dim], -1.0, 1.0))
            self.embedded = tf.nn.embedding_lookup(self.input_embedding, self.train_inputs)

            self.output_embedding = tf.Variable(tf.random_uniform([self.output_count, self.embedding_dim], -1.0, 1.0))
            biases = tf.Variable(tf.zeros([self.output_count]))
            # biases = None

            # self.output_embed = lambda t: tf.nn.embedding_lookup(self.output_embedding, t)
            self.output_embed = lambda t: tf.nn.embedding_lookup(self.input_embedding, t)

            # Compute the average NCE loss for the batch.
            # self.loss = tf.reduce_mean(tf.nn.nce_loss(self.output_embedding, biases,
            #                                           self.train_labels, self.embedded,
            #                                           self.neg_samples, self.output_count))

            combiner = lambda x, y: tf.multiply(x, y)
            self.loss = tf.reduce_mean(custom_2.nce_loss(combiner, self.output_embed, biases,
                                                       self.train_labels, self.embedded,
                                                       self.neg_samples, self.output_count))



            # Construct the SGD optimizer
            self.optimizer = self.optimizer_class(learning_rate=self.learning_rate).minimize(self.loss)

            # Result
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.input_embedding), 1, keep_dims=True))
            self.normalized_embedding = tf.div(self.input_embedding, norm)

    def new_session(self):
        return tf.Session(graph=self.graph)

    def train(self, session, data, max_epoch=None):
        print("Training.....")
        if max_epoch is not None:
            self.max_epoch = max_epoch
        epoch = 0
        step = 0
        average_loss = 0
        while True:
            if epoch >= self.max_epoch:
                break
            if self.max_steps is not None and step >= self.max_steps:
                epoch += 1
                continue
            batch_inputs, batch_labels = data.gen_batch(self.batch_size)
            if batch_inputs is None:
                epoch += 1
                if epoch < self.max_epoch:
                    data.reset()
                    print("")
                continue

            feed_dict = {self.train_inputs : batch_inputs, self.train_labels : batch_labels}
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            
            average_loss += loss_val
            step += 1
            if step % self.checkpoint == 0:
                average_loss /= self.checkpoint
                print("\r", "Epoch ", epoch, " Step ", step, " Loss ", average_loss, end="")
                average_loss = 0

    def write_embeddings(self, session, data):
        ie_result = []
        for i in range(0, self.input_count, self.batch_size):
            feed_dict={self.train_inputs: np.arange(i, i + self.batch_size) % self.input_count}
            ie_result.append(session.run(self.embedded,
                                         feed_dict={self.train_inputs: np.arange(i, i + self.batch_size) % self.input_count}))
        ie_result = np.concatenate(ie_result)[:self.input_count]
        print(ie_result.shape)
        write(self.invec_file, data.input_vocab(), ie_result)
        write(self.outvec_file, data.output_vocab(), self.output_embedding.eval())

        return data.input_vocab, ie_result, data.output_vocab, self.output_embedding.eval()


def train_new():
    trainer = Word2Vec(data_dir="data")
    trainer.embedding_dim = 50
    trainer.checkpoint = 100
    trainer.batch_size = 128
    trainer.learning_rate = 1.0
    data = dataset.TextWindowDataset("data/text8", min_count=5)
    data.num_skips = 4
    data.skip_window = 2
    trainer.build(data)
    with trainer.new_session() as session:
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        trainer.train(session, data, max_epoch=10)
        saver.save(session, "data/word2vec.params")
        trainer.write_embeddings(session, data)
        
    
    
if __name__ == "__main__":
    train_new()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd

from dataset import TextWindowDataset
import custom


def write(filename, index, data):
    df = pd.DataFrame(data)
    df["idx"] = index
    df = df.set_index("idx")
    df.to_csv(filename, header=False, sep="\t")


def vae(x, hidden_dim, latent_dim):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    input_dim = x.shape[-1].value

    W_encoder_input_hidden = weight_variable([input_dim, hidden_dim])
    b_encoder_input_hidden = bias_variable([hidden_dim])

    # Hidden layer encoder
    hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

    W_encoder_hidden_mu = weight_variable([hidden_dim, latent_dim])
    b_encoder_hidden_mu = bias_variable([latent_dim])

    # Mu encoder
    mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

    W_encoder_hidden_logvar = weight_variable([hidden_dim, latent_dim])
    b_encoder_hidden_logvar = bias_variable([latent_dim])

    # Sigma encoder
    logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

    # Sample epsilon
    epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

    # Sample latent variable
    std_encoder = tf.exp(0.5 * logvar_encoder)
    z = mu_encoder + tf.multiply(std_encoder, epsilon)

    W_decoder_z_hidden = weight_variable([latent_dim, hidden_dim])
    b_decoder_z_hidden = bias_variable([hidden_dim])

    # Hidden layer decoder
    hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

    W_decoder_hidden_reconstruction = weight_variable([hidden_dim, input_dim])
    b_decoder_hidden_reconstruction = bias_variable([input_dim])

    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

    x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction

    return x_hat, KLD


class VAEWord2Vec(object):
    def __init__(self):
        # hyper parameters
        self.max_epoch = 1
        self.batch_size = 64
        self.learning_rate = 0.1
        self.checkpoint = 1
        self.max_steps = None
        self.hidden_dim = 500
        self.latent_dim = 50
        self.embedding_dim = 50  # Dimension of the embedding vector.
        self.neg_samples = 16  # Number of negative samples
        self.optimizer_class = tf.train.AdamOptimizer

        # must be specified before building
        self.input_count = None
        self.output_count = None
        self.embed_input = None
        self.embed_output = None

    def build(self, data=None):
        if data is not None:
            self.input_count = data.vocab_size()
            self.output_count = data.vocab_size()
            self.window_size = data.window_size

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size])

            # Construct the variables.
            self.embedding = tf.Variable(tf.random_uniform([self.input_count, self.embedding_dim], -1.0, 1.0))
            biases = tf.Variable(tf.zeros([self.output_count]))
            # biases = None

            self.embedded = tf.reshape(tf.nn.embedding_lookup(self.embedding, self.train_inputs),
                                       [-1, self.embedding_dim * self.window_size])
            encoded, self.KLD = vae(self.embedded, self.hidden_dim, self.latent_dim)
            encoded = tf.reshape(encoded, [-1, self.embedding_dim])

            self.output_embed = lambda t: tf.nn.embedding_lookup(self.embedding, t)
            train_labels = tf.reshape(self.train_inputs, [-1, 1])

            rec_loss = custom.nce_loss(self.output_embed, biases, train_labels,
                                       encoded, self.neg_samples, self.output_count)
            self.rec_loss = tf.reduce_sum(tf.reshape(rec_loss, [-1, self.window_size]), axis=-1)
            loss_div = (1 + self.neg_samples) * self.window_size

            self.loss = tf.reduce_mean(self.rec_loss / loss_div + self.KLD)

            # Construct the SGD optimizer
            self.optimizer = self.optimizer_class(learning_rate=self.learning_rate).minimize(self.loss)

            # Result
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
            self.normalized_embedding = tf.div(self.embedding, norm)

    def new_session(self):
        return tf.Session(graph=self.graph)

    def train(self, session, data, max_epoch=None):
        print("Training.....")
        if max_epoch is not None:
            self.max_epoch = max_epoch
        average_loss = 0
        for epoch in range(self.max_epoch):
            for step, batch in enumerate(data.batches(self.batch_size, equal_batches=True)):
                if self.max_steps is not None and step >= self.max_steps:
                    continue

                feed_dict = {self.train_inputs: batch}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # _, loss_val, rec_loss, KLD = session.run([self.optimizer, self.loss,
                #                                           tf.reduce_mean(self.rec_loss), tf.reduce_mean(self.KLD)],
                #                                          feed_dict=feed_dict)
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

                average_loss += loss_val
                step = step + 1
                if step % self.checkpoint == 0:
                    average_loss /= self.checkpoint
                    print("\r", "Epoch ", epoch, " Step ", step * self.batch_size, " Loss ", average_loss,
                          # "rec ", rec_loss, "KLD ", KLD,
                          end="")
                    average_loss = 0

    def write_embeddings(self, filename, data):
        write(filename, data.vocab(), self.embedding.eval())
        # ie_result = []
        # for i in range(0, self.input_count, self.batch_size):
        #     feed_dict = {self.train_inputs: np.arange(i, i + self.batch_size) % self.input_count}
        #     ie_result.append(session.run(self.embedded,
        #                                  feed_dict={
        #                                      self.train_inputs: np.arange(i, i + self.batch_size) % self.input_count}))
        # ie_result = np.concatenate(ie_result)[:self.input_count]
        # vocab = data.vocab()
        # write(self.invec_file, vocab, ie_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word window VAE')
    parser.add_argument('-v', '--vocab-file', default="data/penn.vocab", type=str, help='vocabulary file')
    parser.add_argument('-c', '--corpus-file', default="data/penn.text", type=str, help='corpus file')
    parser.add_argument('-o', '--output-file', default="data/penn.vectors", type=str, help='corpus file')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('-w', '--window-size', type=int, default=5, help='number of words to encode')
    parser.add_argument('-ed', '--embedding-dim', type=int, default=50, help='embedding dimension')
    parser.add_argument('-hd', '--hidden-dim', type=int, default=400, help='hidden dimension')
    parser.add_argument('-zd', '--z-dim', type=int, default=50, help='latent dimension')
    args = parser.parse_args()

    trainer = VAEWord2Vec()
    trainer.embedding_dim = args.embedding_dim
    trainer.latent_dim = args.z_dim
    trainer.checkpoint = 100
    trainer.batch_size = args.batch_size
    trainer.optimizer_class = tf.train.GradientDescentOptimizer
    trainer.learning_rate = args.learning_rate
    dataset = TextWindowDataset(window_size=args.window_size)
    dataset.mode = "word"
    dataset.one_hot = False
    dataset.reload_vocab(args.vocab_file)
    dataset.load(args.corpus_file)
    trainer.build(dataset)
    with trainer.new_session() as session:
        saver = tf.train.Saver()
        tf.initialize_all_variables().run()
        trainer.train(session, dataset, max_epoch=args.epochs)
        saver.save(session, "data/word2vec_bias.params")
        trainer.write_embeddings(args.output_file, dataset)
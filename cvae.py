import argparse
import numpy as np
import torch
import torch.nn as nn
import visdom
from torch.autograd import Variable

from dataset import TextWindowDataset

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_zeros, ng_ones


fudge = 1e-7


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.view(-1, self.input_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.input_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        # fixing numerical instabilities of sigmoid with a fudge
        mu_img = (self.sigmoid(self.fc21(hidden))+fudge) * (1-2*fudge)
        return mu_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, input_dim, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(input_dim, z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # setup hyperparameters for prior p(z)
        # the type_as ensures we get cuda Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        # decode the latent code z
        mu_img = self.decoder.forward(z)
        # score against actual images
        pyro.observe("obs", dist.bernoulli, x.view(-1, self.input_dim), mu_img)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder.forward(x)
        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_mu, z_sigma = self.encoder(x)
        # sample in latent space
        z = dist.normal(z_mu, z_sigma)
        # decode the image (note we don't sample in image space)
        mu_img = self.decoder(z)
        return mu_img

    def model_sample(self, batch_size=1):
        # sample the handwriting style from the constant prior distribution
        prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
        prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
        zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)
        mu = self.decoder.forward(zs)
        xs = pyro.sample("sample", dist.bernoulli, mu)
        return xs, mu

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-v', '--vocab-file', default="data/penn.vocab", type=str, help='vocabulary file')
    parser.add_argument('-c', '--corpus-file', default="data/penn.text", type=str, help='corpus file')
    parser.add_argument('-o', '--output-file', default="data/penn.vectors", type=str, help='corpus file')
    parser.add_argument('-n', '--num-epochs', default=1, type=int, help='number of training epochs')
    parser.add_argument('-bs', '--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    dataset = TextWindowDataset(window_size=5)
    dataset.reload_vocab(args.vocab_file)
    dataset.load(args.corpus_file)
    input_dim = int(np.product(dataset.dim()))
    print(dataset.size(), dataset.vocab_size(), dataset.alphabet_size(), dataset.word_length)

    # setup the VAE
    vae = VAE(input_dim, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

    train_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in dataset.batches(args.batch_size, equal_batches=False):
            x = torch.from_numpy(x).float()
            if args.cuda:
                x = x.cuda()
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            loss = svi.step(x)
            epoch_loss += loss
            del(loss)

        # report training diagnostics
        normalizer_train = dataset.size()
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    result = []
    for x in dataset.word_batches(args.batch_size):
        x = torch.from_numpy(x).float()
        if args.cuda:
            x = x.cuda()
        x = Variable(x)
        z = vae.encoder(x).cpu().data.numpy()
        result.append(z)
    result = np.vstack(result)

    with open(args.output_file, "w", encoding="utf-8") as target:
        for i in range(dataset.vocab_size()):
            target.write(dataset.index_to_word[i] + "\t" + "\t".join([str(x) for x in result[i]]) + "\n")

    return vae


if __name__ == '__main__':
    model = main()

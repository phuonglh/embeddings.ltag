from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from dataset import TextWindowDataset

import time

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_dim))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= args.batch_size * self.input_dim

        return BCE + KLD

    def get_optimizer(self, lr=1e-3):
        return optim.Adam(self.parameters(), lr=lr)


def train(model, epoch, dataset, args):
    model.train()
    train_loss = 0
    optimizer = model.get_optimizer()
    timer = time.time()
    for batch_idx, data in enumerate(dataset.batches(args.batch_size)):
        data = Variable(torch.from_numpy(data).float())
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAverage batch time: {:.3f}'
                  .format(epoch,
                          batch_idx * len(data),
                          dataset.size(),
                          100. * batch_idx / dataset.size(),
                          loss.data[0] / len(data),
                          (time.time() - timer) / (batch_idx + 1)),
                  end="")

    print('====> Epoch: {} Average loss: {:.4f} Time: {:.3f}'
          .format(epoch, train_loss / dataset.size(), time.time() - timer))


def get_codes(model, dataset, args):
    result = []
    for x in dataset.word_batches(args.batch_size):
        x = torch.from_numpy(x).float()
        if args.cuda:
            x = x.cuda()
        x = Variable(x)
        z = model.encode(x.view(-1, model.input_dim)).cpu().data.numpy()
        result.append(z)
    return np.vstack(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('-v', '--vocab-file', default="data/penn.vocab", type=str, help='vocabulary file')
    parser.add_argument('-c', '--corpus-file', default="data/penn.text", type=str, help='corpus file')
    parser.add_argument('-o', '--output-file', default="data/penn.vectors", type=str, help='corpus file')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('-hd', '--hidden-dim', type=int, default=500, help='hidden dimension')
    parser.add_argument('-zd', '--z-dim', type=int, default=50, help='latent dimension')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    dataset = TextWindowDataset(window_size=5)
    dataset.reload_vocab(args.vocab_file)
    dataset.load(args.corpus_file)
    input_dim = int(np.product(dataset.dim()))
    print(dataset.size(), dataset.vocab_size(), dataset.alphabet_size(), dataset.word_length)

    model = VAE(input_dim, args.hidden_dim, args.z_dim)
    if args.cuda:
        model.cuda()
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, dataset, args=args)

    with open(args.output_file, "w", encoding="utf-8") as target:
        result = get_codes(model, dataset, args=args)
        for i in range(dataset.vocab_size()):
            target.write(dataset.index_to_word[i] + "\t" + "\t".join([str(x) for x in result[i]]) + "\n")

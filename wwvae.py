from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Parameter, functional as F, CrossEntropyLoss, Embedding

from dataset import TextWindowDataset, to_one_hot

import time


class WordWindowVAE(nn.Module):
    def __init__(self, window_size, input_dim, embedding_dim, hidden_dim, z_dim):
        super(WordWindowVAE, self).__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.flat_dim = window_size * self.embedding_dim

        self.embedding = Embedding(self.input_dim, self.embedding_dim)
        self.logit_bias = Parameter(torch.zeros(self.input_dim))

        self.fc1 = nn.Linear(self.flat_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.flat_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.recon_loss = CrossEntropyLoss()

    def embed(self, x):
        return self.embedding(x)

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
        return self.fc4(h3)

    def logits(self, x):
        result = torch.bmm(x.view(x.size(0), -1, self.embedding_dim),
                           self.embedding.weight.t().expand(x.size(0), self.embedding_dim, self.input_dim))
        return result + self.logit_bias.expand_as(result)

    def forward(self, x):
        mu, logvar = self.encode(self.embed(x).view(-1, self.flat_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # BCE = self.recon_loss(input=self.logits(recon_x).view(-1, self.input_dim), target=x.view(-1))

        target = Variable(torch.zeros(x.numel(), self.input_dim))
        if torch.cuda.is_available():
            target = target.cuda()
        target.scatter_(1, x.view(-1, 1), 1)
        BCE = F.binary_cross_entropy_with_logits(input=self.logits(recon_x).view(-1, self.input_dim),
                                                 target=target)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.numel()

        return BCE + KLD

    def get_optimizer(self, lr=1e-3):
        return optim.Adam(self.parameters(), lr=lr)


def train(model, epoch, dataset, args):
    model.train()
    train_loss = 0
    optimizer = model.get_optimizer()
    timer = time.time()

    for batch_idx, batch in enumerate(dataset.batches(args.batch_size)):
        # data = to_one_hot(batch.reshape(-1), dataset.vocab_size()).reshape(batch.shape + (-1,))
        # data = Variable(torch.from_numpy(data).float())
        batch = Variable(torch.from_numpy(batch).long())
        if args.cuda:
            # data = data.cuda()
            batch = batch.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = model.loss_function(recon_batch, batch, mu, logvar)
        # recon = model.embed(batch[:, model.window_size // 2].unsqueeze(-1).repeat(1, model.window_size))
        # loss = model.recon_loss(model.logits(recon).view(-1, model.input_dim), batch.view(-1))
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % 10 == 0: 
            print('\r' + " " * 100, end="")
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tAverage batch time: {:.3f}\tLoss: {}'
                  .format(epoch,
                          batch_idx * len(batch),
                          dataset.size(),
                          100. * batch_idx / dataset.size(),
                          (time.time() - timer) / (batch_idx + 1),
                          loss.data[0] / len(batch),),
                  end="")

    print('\n====> Epoch: {} Average loss: {:.4f} Time: {:.3f}'
          .format(epoch, train_loss / dataset.size(), time.time() - timer))


def get_codes(model, dataset, args):
    result = []
    for x in dataset.vocab_batches(args.batch_size):
        # x = to_one_hot(x.reshape(-1), dataset.vocab_size()).reshape(x.shape + (-1,))
        x = Variable(torch.from_numpy(x).long())
        if args.cuda:
            x = x.cuda()
        z = model.embed(x).view(-1, model.embedding_dim).cpu().data.numpy()
        result.append(z)
    return np.vstack(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word window VAE')
    parser.add_argument('-v', '--vocab-file', default="data/penn.vocab", type=str, help='vocabulary file')
    parser.add_argument('-c', '--corpus-file', default="data/penn.text", type=str, help='corpus file')
    parser.add_argument('-o', '--output-file', default="data/penn.vectors", type=str, help='corpus file')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('-w', '--window-size', type=int, default=5, help='number of words to encode')
    parser.add_argument('-ed', '--embedding-dim', type=int, default=50, help='embedding dimension')
    parser.add_argument('-hd', '--hidden-dim', type=int, default=400, help='hidden dimension')
    parser.add_argument('-zd', '--z-dim', type=int, default=50, help='latent dimension')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=2, help='how many batches to wait before logging')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    dataset = TextWindowDataset(window_size=args.window_size)
    dataset.mode = "word"
    dataset.one_hot = False
    dataset.reload_vocab(args.vocab_file)
    dataset.load(args.corpus_file)
    input_dim = int(np.product(dataset.dim()))
    print("Diagnostics: ", args.cuda, dataset.size(), dataset.vocab_size(), dataset.alphabet_size(), dataset.word_length)

    model = WordWindowVAE(window_size=dataset.window_size,
                          input_dim=dataset.vocab_size(),
                          embedding_dim=args.embedding_dim,
                          hidden_dim=args.hidden_dim,
                          z_dim=args.z_dim)
    if args.cuda:
        model.cuda()
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, dataset, args=args)
        if epoch % args.log_interval == 0:
            with open(args.output_file.replace(".", "." + str(epoch) + "."), "w", encoding="utf-8") as target:
                result = get_codes(model, dataset, args=args)
                for i in range(dataset.vocab_size()):
                    target.write(dataset.index_to_word[i] + "\t" + "\t".join([str(x) for x in result[i]]) + "\n")

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
import ml_metrics
import codecs
import time

import framenet

def normalize(vectors):
    for name, vector in vectors.iteritems():
        vectors[name] = vector / np.linalg.norm(vector)


def random_sphere(vocab, dim):
    result = dict(zip(vocab, np.random.normal(size=(len(vocab), dim))))
    normalize(result)
    return result


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def save_vectors(vectors, filename):
    pd.DataFrame.from_dict(vectors, orient="index").to_csv(filename, sep="\t", header=False, encoding="utf-8")


def cosine(vectors, pairs, normal=False):
    sim = np.dot if normal else cosine
    return [sim(vectors[x], vectors[y]) for x, y in pairs]


def mean_cosine(vectors, pairs, normal=False):
    return np.mean(cosine(vectors, pairs, normal))


def load_pairs(filename):
    return pd.read_csv(filename, sep="\t", header=None, encoding="utf-8").as_matrix()

def list_vectors(filename, delimiter="\t", lower=False):
    data = pd.read_csv(filename, sep="\t", header=None, encoding="utf-8")
    if lower:
        data[0] = data[0].str.lower()
    return data[0].tolist(), data[data.columns[1:]].as_matrix()

def load_vectors(filename, delimiter="\t", lower=False):
    x, y = list_vectors(filename, delimiter, lower)
    return dict(zip(x, y))

def nearest(vectors, word, normal=False):
    sim = np.dot if normal else cosine
    sims = [(x, sim(vectors[word], v)) for x, v in vectors.iteritems() if x != word]
    return sorted(sims, key=lambda x: x[1], reverse=True)


def test_mean_cosine(filename):
    same = load_pairs("data/same.txt")
    control = load_pairs("data/control.txt")
    vectors = load_vectors(filename)
    normalize(vectors)
    print mean_cosine(vectors, same, normal=True), mean_cosine(vectors, control, normal=True)

def test_ks(filename):
    vectors = load_vectors(filename)
    normalize(vectors)
    same = cosine(vectors, load_pairs("data/same.txt"), normal=True)
    control = cosine(vectors, load_pairs("data/control.txt"), normal=True)
    print ks_2samp(np.array(same), np.array(control))


def test_ranking(filename):
    vectors = load_vectors(filename)
    normalize(vectors)
    with codecs.open("data/rank.txt", "r", encoding="utf-8") as source:
        lines = [line.strip().split() for line in source.readlines()]
    actual = [line[1:] for line in lines]
    predicted = [[x[0] for x in nearest(vectors, line[0], normal=True)] for line in lines]
    print ml_metrics.mapk(actual, predicted, k=100)


def test_ranking_2(filename):
    vectors = load_vectors(filename)
    normalize(vectors)
    with codecs.open("data/rank.txt", "r", encoding="utf-8") as source:
        lines = [line.strip().split() for line in source.readlines()]
    vocab = set([word for line in lines for word in line])
    vectors = {word: vector for word, vector in vectors.iteritems() if word in vocab}
    actual = [line[1:] for line in lines]
    predicted = [[x[0] for x in nearest(vectors, line[0], normal=True)] for line in lines]
    print ml_metrics.mapk(actual, predicted, k=100)


if __name__ == "__main__":
    # filenames = ["data/penn/word2vec_1/invec.txt", "data/penn/word2vec_2/invec.txt"]
    filenames = ["data/words.None.169"]
    tests = [test_mean_cosine, test_ranking, test_ranking_2, test_ks]

    for filename in filenames:
        print filename
        for test in tests:
            timer = time.time()
            print test.__name__,
            test(filename)
            print "time =", time.time() - timer
        print "\n"

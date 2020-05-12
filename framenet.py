import xml.etree.ElementTree as ElementTree
from collections import defaultdict
import codecs

import pandas as pd
import time


def parse_lu():
    frames = defaultdict(list)
    root = ElementTree.parse("data/luIndex.xml").getroot()
    for x in root.getchildren()[1:]:
        if x.attrib["status"] != "Created" or " " in x.attrib["name"]:
            continue
        word, kind = x.attrib["name"].split(".")
        frames[x.attrib["frameID"]].append((word, kind))
    with codecs.open("data/lu.txt", "w", encoding="utf-8") as output:
        for id, frame in frames.iteritems():
            for word, kind in frame:
                output.write("\t".join([id, word, kind]) + "\n")


def gen_lu_pairs(word_files=["data/words.txt"], size=10000):
    lu = pd.read_csv("data/lu.txt", sep="\t", encoding="utf-8")
    for word_file in word_files:
        vocab = pd.read_csv(word_file, sep="\t", header=None, encoding="utf-8")
        lu = lu[lu.word.isin(vocab[0])]
    counts = lu.groupby("frame").word.nunique()
    frames = counts[counts > 1].index
    lu_plus = lu[lu.frame.isin(frames)]
    same = []
    control = []
    timer = time.time()
    for i in range(size):
        control.append(lu.sample(2).word.tolist())

        x = lu_plus.sample().iloc[0]
        y = x
        while y.word == x.word:
            y = lu[lu.frame == x.frame].sample().iloc[0]
        same.append([x.word, y.word])
        print (i+1), (time.time() - timer) / float(i + 1)
    pd.DataFrame(same).to_csv("data/same.txt", sep="\t", header=False, index=False, encoding="utf-8")
    pd.DataFrame(control).to_csv("data/control.txt", sep="\t", header=False, index=False, encoding="utf-8")


def gen_frame_rankings(word_files=["data/words.txt"], size=1000):
    lu = pd.read_csv("data/lu.txt", sep="\t", encoding="utf-8")
    for word_file in word_files:
        vocab = pd.read_csv(word_file, sep="\t", header=None, encoding="utf-8")
        lu = lu[lu.word.isin(vocab[0])]
    counts = lu.groupby("frame").word.nunique()
    frames = counts[counts > 1].index
    lu_plus = lu[lu.frame.isin(frames)]
    result = []
    timer = time.time()
    seeds = lu_plus.sample(size)
    for i in range(size):
        x = seeds.iloc[i]
        result.append([x.word] + [y for y in lu[lu.frame == x.frame].word.unique() if y != x.word])
        print (i+1), (time.time() - timer) / float(i + 1)
    with codecs.open("data/rank.txt", "w", encoding="utf-8") as output:
        output.writelines(["\t".join(line) + "\n" for line in result])


def filter_word_files(word_files=["data/words.txt"]):
    lu = pd.read_csv("data/lu.txt", sep="\t", encoding="utf-8").word.unique()
    vocab = {}
    for word_file in word_files:
        vocab[word_file] = pd.read_csv(word_file, sep="\t", header=None, encoding="utf-8")
        lu = [x for x in lu if x in vocab[word_file][0].values]
    for word_file, data in vocab.items():
        data[data[0].isin(lu)].to_csv(word_file, sep="\t", header=False, index=False, encoding="utf-8")

def filter_word_files_2(word_files=["data/words.txt"]):
    lu = pd.read_csv(word_files[0], sep="\t", header=None, encoding="utf-8")[0].values
    for word_file in word_files[1:]:
        data = pd.read_csv(word_file, sep="\t", header=None, encoding="utf-8")
        data[data[0].isin(lu)].to_csv(word_file, sep="\t", header=False, index=False, encoding="utf-8")
        print word_file


if __name__ == "__main__":
    gen_lu_pairs(word_files=["data/invec.txt"])
    gen_frame_rankings(word_files=["data/invec.txt"])
    # filter_word_files(["data/penn/word2vec_1/invec.txt", "data/penn/word2vec_2/invec.txt"])
    # filter_word_files_2(["data/penn/word2vec_2/invec.txt"] + ["data/penn/model_1/words.txt." + str(n) for n in range(117, 118)])

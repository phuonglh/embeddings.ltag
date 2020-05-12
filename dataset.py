import numpy as np
import pickle
from collections import Counter
import os


def build_dictionary(filenames, vocab_size):
    word_counts = Counter()
    char_set = set()
    max_length = 0
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as source:
            for line in source:
                words = line.split()
                word_counts.update(words)
                char_set.update(line)
                max_length = max(max_length, len(words))
    return ({w[0]: i + 1 for i, w in enumerate(word_counts.most_common(vocab_size))},
            {c: i + 1 for i, c in enumerate(char_set)},
            max_length)


# removes zeroth column
def to_one_hot(rows, dim):
    result = np.zeros((rows.shape[0], dim))
    result[np.arange(rows.shape[0]), rows] = 1
    result[:, 0] = 0
    return result


class TextWindowDataset(object):
    def __init__(self, window_size, one_hot=True, save_mem=False,unk="<UNK>"):
        self.window_size = window_size
        self.one_hot = one_hot
        self.save_mem = save_mem
        self.index = 0
        self.unk = unk
        self.data = []
        self.mode = "concat"

    def vocab(self):
        return [self.index_to_word[i] for i in range(self.vocab_size())]

    def vocab_size(self):
        return len(self.word_to_index) + 1

    def alphabet_size(self):
        return len(self.char_to_index) + 1

    def dim(self):
        if self.one_hot:
            return self.window_size, self.one_hot_table.shape[-1]
        else:
            return self.window_size, self.table.shape[-1]

    def size(self):
        return len(self.data)

    def text(self, line):
        return " ".join(self.index_to_word.get(x, self.unk) for x in line)

    def code(self, word):
        return [self.word_to_index.get(word, 0)] + self.word_to_char[word]

    def pad(self, row):
        if len(row) < self.word_length + 1:
            return row + [0] * (self.word_length + 1 - len(row))
        else:
            return row[:self.word_length + 1]

    def padded_to_one_hot(self, code):
        result = [to_one_hot(code[:, 0], self.vocab_size())]
        for i in range(self.word_length):
            result.append(to_one_hot(code[:, i + 1], self.alphabet_size()))
        return np.hstack(result)

    def build_tables(self):
        self.table = np.vstack([self.pad(self.code(self.index_to_word.get(i, self.unk)))
                                for i in range(self.vocab_size())])
        if self.one_hot and not self.save_mem:
            self.one_hot_table = self.padded_to_one_hot(self.table)

    def load_vocab(self, filenames, vocab_size):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.word_to_index, self.char_to_index, self.input_length = \
            build_dictionary(filenames, vocab_size)
        self.index_to_word = {w: i for i, w in self.word_to_index.items()}
        self.index_to_word[0] = self.unk
        self.index_to_char = {c: i for i, c in self.char_to_index.items()}
        self.word_length = max([len(w) for w in self.word_to_index])
        self.word_to_char = {w: [self.char_to_index[c] for c in w] for w in self.word_to_index}
        self.word_to_char[self.unk] = []
        self.build_tables()

    def save_vocab(self, filename):
        with open(filename, "wb") as target:
            pickle.dump((self.word_to_index, self.index_to_word,
                         self.char_to_index, self.index_to_char,
                         self.input_length, self.word_length,
                         self.word_to_char),
                        target)

    def reload_vocab(self, filename):
        with open(filename, "rb") as source:
            (self.word_to_index, self.index_to_word,
             self.char_to_index, self.index_to_char,
             self.input_length, self.word_length,
             self.word_to_char) = pickle.load(source)
        self.build_tables()

    def batch_concat(self, batch_size):
        result = self.data[self.index: self.index + batch_size]
        if self.one_hot:
            if self.save_mem:
                result = self.padded_to_one_hot(self.table[result])
            else:
                result = self.one_hot_table[result].copy()
        else:
            result = self.table[result].copy()
        self.index += batch_size
        return result

    def batch_word(self, batch_size):
        result = self.data[self.index: self.index + batch_size]
        if self.one_hot:
            if self.save_mem:
                result = np.array([to_one_hot(row, self.vocab_size()) for row in result])
            else:
                result = self.one_hot_table[result, :self.vocab_size()]
        self.index += batch_size
        return result

    def batches(self, batch_size, equal_batches=False):
        if self.mode == "concat":
            get_batch = self.batch_concat
        elif self.mode == "word":
            get_batch = self.batch_word

        alive = True
        while alive:
            result = get_batch(batch_size)
            alive = len(result) == batch_size
            if len(result) == batch_size or (not equal_batches and len(result) > 0):
                yield result
        self.reset()

    def __getitem__(self, index):
        result = self.data[self.index]
        if self.one_hot:
            if self.save_mem:
                result = self.padded_to_one_hot(self.table[result])
            else:
                result = self.one_hot_table[result].copy()
        else:
            result = self.table[result].copy()
        return result

    def vocab_batches(self, batch_size):
        index = 0
        while index < self.vocab_size():
            if self.one_hot:
                result = self.one_hot_table[index: index + batch_size]
            elif self.mode == "concat":
                result = self.table[index: index + batch_size]
            elif self.mode == "word":
                result = np.arange(index, min(index + batch_size, self.vocab_size()))
            index += batch_size
            yield result

    def reset(self):
        self.index = 0

    def load(self, filename):
        self.data = []
        with open(filename, "r", encoding="utf-8") as source:
            lines = [[self.word_to_index.get(word, 0) for word in line.split()] for line in source]
        for line in lines:
            self.data.extend([line[i: i + self.window_size] for i in range(len(line) - self.window_size + 1)])
        self.data = np.array(self.data)


if __name__ == "__main__":
    folder = "billion/original/training/"
    vocab_file = "data/penn.vocab"
    # files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    files = ["data/penn.text"]
    data = TextWindowDataset(4, save_mem=False)
    data.reload_vocab(vocab_file)
    # data.load_vocab(files, 11423)
    data.load(files[0])
    print(files[0], max([len(w) for w in data.word_to_index]), data.word_length, data.alphabet_size())
    data.save_vocab(vocab_file)
    data.mode = "word"
    for x in data.batches(5, True):
        print(data.index_to_word[np.where(x[0][0] == 1)[0][0]])
    print()
    for x in data.batches(5, False):
        print(data.index_to_word[np.where(x[0][0] == 1)[0][0]])
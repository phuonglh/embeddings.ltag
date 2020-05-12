import operator
import collections
from random import randint
import codecs

import numpy as np
from sklearn.model_selection import StratifiedKFold

from vectors import list_vectors, load_vectors, normalize
from keras.utils import np_utils

def split_file(filename, fractions, shuffle=False):
    with codecs.open(filename, "r", encoding="utf-8") as source:
        lines = map(lambda x: x if x.endswith("\n") else x + "\n", source.readlines())
    if shuffle:
        np.random.shuffle(lines)
    indices = np.append([0], np.rint(np.cumsum(fractions) * len(lines)).astype(int))
    for i in range(len(fractions)):
        with codecs.open(filename + "." + str(i), "w", encoding="utf-8") as target:
            target.writelines(lines[indices[i]: indices[i + 1]])

def split_cv(filename, n, shuffle=False, perm=None):
    with codecs.open(filename, "r", encoding="utf-8") as source:
        lines = map(lambda x: x if x.endswith("\n") else x + "\n", source.readlines())
    if perm is not None:
        lines = [lines[i] for i in perm if i < len(lines)]
    elif shuffle:
        np.random.shuffle(lines)
    indices = np.rint([i / float(n) * len(lines) for i in range(n + 1)]).astype(int)
    for i in range(n):
        temp = lines[:indices[i]] + lines[indices[i + 1]:]
        with codecs.open(filename + ".train." + str(i), "w", encoding="utf-8") as target:
            target.writelines(temp[:indices[n - 2]])
        with codecs.open(filename + ".valid." + str(i), "w", encoding="utf-8") as target:
            target.writelines(temp[indices[n - 2]:])
        with codecs.open(filename + ".test." + str(i), "w", encoding="utf-8") as target:
            target.writelines(lines[indices[i]: indices[i + 1]])


def strat_cv(filename, n, perm, shuffle=False):
    with codecs.open(filename, "r", encoding="utf-8") as source:
        lines = map(lambda x: x if x.endswith("\n") else x + "\n", source.readlines())
    if shuffle:
        np.random.shuffle(lines)
    if perm is None:
        X = []
        y = []
        for line in lines:
            text, label = line.split("\t")
            X.append(text)
            y.append(label)
        skf = StratifiedKFold(n_splits=n)
        perm = [t[1] for t in skf.split(X, y)]
    for i in range(n):
        with codecs.open(filename + ".train." + str(i), "w", encoding="utf-8") as target:
            target.writelines([lines[x] for k in range(n) for x in perm[k] if k != i and k != (i + 1) % n])
        with codecs.open(filename + ".valid." + str(i), "w", encoding="utf-8") as target:
            target.writelines([lines[x] for x in perm[(i + 1) % n]])
        with codecs.open(filename + ".test." + str(i), "w", encoding="utf-8") as target:
            target.writelines([lines[x] for x in perm[i]])
    return perm


class EmbeddedTextDataset(object):
    def __init__(self, filename, input_length=None, num_classes=None, delimiter="\t", lower=False, normalized=False):
        self.index = 0
        self.input_length = input_length
        self.num_classes = num_classes
        self.vocab = load_vectors(filename, delimiter, lower)
        if normalized:
            normalize(self.vocab)
        self.embedding_dim = self.vocab["UNK"].shape[0]
        self.x = []
        self.y = []

    def size(self):
        return len(self.x)

    def zero_unk(self):
        self.vocab["UNK"] = np.zeros((self.embedding_dim))

    def read(self, filename, delimiter="\t"):
        input_length = 0
        with codecs.open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text, label = line.split("\t")
                self.x.append(np.array([self.vocab.get(token, self.vocab["UNK"]) for token in text.split()]))
                self.y.append(int(label))
                input_length = max(input_length, len(text))
        if self.input_length is None:
            self.input_length = input_length
        if self.num_classes is None:
            self.num_classes = len(set(self.y))

    def batch_x(self, begin, end):
        x = []
        for i in xrange(begin, end):
            if self.x[i].shape[0] > self.input_length:
                x.append(self.x[i][:self.input_length])
            else:
                x.append(np.pad(self.x[i],
                                ((0, self.input_length - self.x[i].shape[0]), (0, 0)),
                                mode="constant",
                                constant_values=0))
        return np.array(x)

    def batch_y(self, begin, end):
        return np_utils.to_categorical(self.y[begin:end], self.num_classes)

    def all_x(self):
        return self.batch_x(0, self.size())

    def all_y(self):
        return self.batch_y(0, self.size())

    def gen_batch(self, batch_size):
        if self.index + batch_size < self.size():
            result = self.batch(self.index, self.index + batch_size)
            self.index += batch_size
        else:
            result = None
            self.index = 0
        return result


class TextDataset(EmbeddedTextDataset):
    def __init__(self, filename, input_length=None, num_classes=None, delimiter="\t", lower=False):
        self.index = 0
        self.input_length = input_length
        self.num_classes = num_classes
        self.vocab = list_vectors(filename, delimiter, lower)[0]
        self.vocab = dict(zip(self.vocab, range(len(self.vocab))))
        self.x = []
        self.y = []

    def vocab_size(self):
        return len(self.vocab)

    def read(self, filename, delimiter="\t"):
        input_length = 0
        with codecs.open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text, label = line.split("\t")
                self.x.append(np.array([self.vocab.get(token, 0) for token in text.split()]))
                self.y.append(int(label))
                input_length = max(input_length, len(text))
        if self.input_length is None:
            self.input_length = input_length
        if self.num_classes is None:
            self.num_classes = len(set(self.y))

    def batch_x(self, begin, end):
        x = []
        for i in xrange(begin, end):
            if self.x[i].shape[0] > self.input_length:
                x.append(self.x[i][:self.input_length])
            else:
                x.append(np.pad(self.x[i],
                                ((0, self.input_length - self.x[i].shape[0])),
                                mode="constant",
                                constant_values=0))
        return np.array(x)

class EmbeddedTemplateDataset(EmbeddedTextDataset):
    def __init__(self, template_file, vocab_file, input_length=None, num_classes=None, delimiter="\t", lower=False,
                 normalized=False):
        self.index = 0
        self.input_length = input_length
        self.num_classes = num_classes
        self.template = load_vectors(template_file, delimiter, lower)
        self.vocab = load_vectors(vocab_file, delimiter, lower)
        self.embedding_dim = self.template["UNK"].shape[0] + self.vocab["UNK"].shape[0]
        self.x = []
        self.y = []

    def zero_unk(self):
        self.vocab["UNK"] = np.zeros(self.vocab["UNK"].shape)
        self.template["UNK"] = np.zeros(self.template["UNK"].shape)

    def read(self, filename, delimiter="\t"):
        input_length = 0
        with codecs.open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text, label = line.split("\t")

                def lookup(token):
                    template, word = token.rsplit("/", 1)
                    return np.hstack((self.template.get(template, self.template["UNK"]), self.vocab.get(word, self.vocab["UNK"])))

                self.x.append(np.array([lookup(token) for token in text.split()]))
                self.y.append(int(label))
                input_length = max(input_length, len(text))
        if self.input_length is None:
            self.input_length = input_length
        if self.num_classes is None:
            self.num_classes = len(set(self.y))


class TemplateDataset(EmbeddedTemplateDataset):
    def __init__(self, template_file, vocab_file, input_length=None, num_classes=None, delimiter="\t", lower=False):
        self.index = 0
        self.input_length = input_length
        self.num_classes = num_classes
        self.template = list_vectors(template_file, delimiter, lower)[0]
        self.template = dict(zip(self.template, range(len(self.template))))
        self.vocab = list_vectors(vocab_file, delimiter, lower)[0]
        self.vocab = dict(zip(self.vocab, range(len(self.vocab))))
        self.x = []
        self.y = []

    def num_templates(self):
        return len(self.template)

    def vocab_size(self):
        return len(self.vocab)

    def read(self, filename, delimiter="\t"):
        input_length = 0
        with codecs.open(filename, "r", encoding="utf-8") as source:
            for line in source:
                text, label = line.split("\t")

                def lookup(token):
                    template, word = token.rsplit("/", 1)
                    return np.hstack((self.template.get(template, 0), self.vocab.get(word, 0)))

                self.x.append(np.array([lookup(token) for token in text.split()]))
                self.y.append(int(label))
                input_length = max(input_length, len(text))
        if self.input_length is None:
            self.input_length = input_length
        if self.num_classes is None:
            self.num_classes = len(set(self.y))

    def batch_x(self, begin, end):
        x = []
        for i in xrange(begin, end):
            if self.x[i].shape[0] > self.input_length:
                x.append(self.x[i][:self.input_length])
            else:
                x.append(np.pad(self.x[i],
                                ((0, self.input_length - self.x[i].shape[0]), (0, 0)),
                                mode="constant",
                                constant_values=0))
        x = np.array(x)
        return [x[:, :, 0], x[:, :, 1]]


class TextWindowDataset(object):
    def read_data(self, filename):
      with open(filename, "r") as f:
        data = f.read().split()
      return data

    def build_dataset(self, words, min_count):
        count = [['UNK', -1]]
        vocab = [(word, frequency) for word, frequency in collections.Counter(words).items() if frequency > min_count]
        count.extend(sorted(vocab, key=lambda x: x[1], reverse=True))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                  index = 0  # dictionary['UNK']
                  unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary

    def __init__(self, filename, min_count=10):
        self.num_skips = 2
        self.skip_window = 1
        words = self.read_data(filename)
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset(words, min_count)
        del words  # Hint to reduce memory.
        self.data_index = 0

    def input_vocab(self):
        return np.array([x[0] for x in sorted(self.dictionary.items(), key=operator.itemgetter(1))])

    def output_vocab(self):
        return self.input_vocab()

    def input_count(self):
        return len(self.count)

    def output_count(self):
        return len(self.count)

    def reset(self):
        self.data_index = 0

    # Step 3: Function to generate a training batch for the skip-gram model.
    def gen_batch(self, batch_size):
        if self.data_index + (batch_size // self.num_skips) * self.skip_window > len(self.data):
            return None, None
        assert batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

        
class TextFilesDataset(object):
    def read_data(self, filename):
      with open(filename, "r") as f:
        data = f.read().split()
      return data

    def build_dict(self, min_count):
        count = [['UNK', -1]]
        vocab = [(word, frequency) for word, frequency in self.counter.items() if frequency > min_count]
        count.extend(sorted(vocab, key=lambda x: x[1], reverse=True))
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
            
    def load_file(self, filename):
        words = self.read_data(filename)
        self.data = list()
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                  index = 0  # dictionary['UNK']
            self.data.append(index)
        self.data_index = 0

    def __init__(self, filenames, min_count=10):
        self.num_skips = 2
        self.skip_window = 1
        self.counter = collections.Counter()
        for filename in filenames:
            self.counter.update(self.read_data(filename))
        self.build_dict(min_count)

    def input_vocab(self):
        return np.array([x[0] for x in sorted(self.dictionary.items(), key=operator.itemgetter(1))])

    def output_vocab(self):
        return self.input_vocab()

    def input_count(self):
        return len(self.dictionary)

    def output_count(self):
        return len(self.dictionary)

    def reset(self):
        self.data_index = 0

    # Step 3: Function to generate a training batch for the skip-gram model.
    def gen_batch(self, batch_size):
        if self.data_index + (batch_size // self.num_skips) * self.skip_window > len(self.data):
            return None, None
        assert batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels


class TaggedTextFilesDataset(object):
    def read_data(self, filename):
        with open(filename, "r") as f:
            result = []
            for line in f:
                result.extend(line.strip("\n").split())
        return result

    def build_dict(self, min_count):
        count = [['UNK', -1]]
        vocab = [(word, frequency) for word, frequency in self.counter.items() if frequency > min_count and word != ""]
        count.extend(sorted(vocab, key=lambda x: x[1], reverse=True))
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def build_tags(self):
        count = [['UNK', -1]]
        vocab = [(word, frequency) for word, frequency in self.tag_counter.items()]
        count.extend(sorted(vocab, key=lambda x: x[1], reverse=True))
        self.tag_dictionary = dict()
        for word, _ in count:
            self.tag_dictionary[word] = len(self.tag_dictionary)
        self.reverse_tag_dictionary = dict(zip(self.tag_dictionary.values(), self.tag_dictionary.keys()))

    def load_file(self, filename):
        words = self.read_data(filename)
        tags = self.read_data(filename.replace(".txt", ".out"))
        assert len(words) == len(tags)
        self.data = list()
        for word, tag in zip(words, tags):
            if word == "":
                continue
            if word in self.dictionary:
                word_index = self.dictionary[word]
            else:
                word_index = 0  # dictionary['UNK']
            if tag in self.tag_dictionary:
                tag_index = self.tag_dictionary[tag]
            else:
                tag_index = 0
            self.data.append((word_index, tag_index))
        self.data_index = 0

    def __init__(self, filenames, min_count=10):
        self.num_skips = 2
        self.skip_window = 1
        self.counter = collections.Counter()
        self.tag_counter = collections.Counter()
        for filename in filenames:
            words = self.read_data(filename)
            tags = self.read_data(filename.replace(".txt", ".out"))
            assert len(words) == len(tags)
            self.counter.update(words)
            self.tag_counter.update(tags)
        self.build_dict(min_count)
        self.build_tags()

    def input_vocab(self):
        return np.array([x[0] for x in sorted(self.dictionary.items(), key=operator.itemgetter(1))])

    def tag_vocab(self):
        return np.array([x[0] for x in sorted(self.tag_dictionary.items(), key=operator.itemgetter(1))])

    def output_vocab(self):
        return self.input_vocab()

    def input_count(self):
        return len(self.dictionary)

    def tag_count(self):
        return len(self.tag_dictionary)

    def output_count(self):
        return len(self.dictionary)
        
    def size(self):
        return len(self.data)

    def reset(self):
        self.data_index = 0

    # Step 3: Function to generate a training batch for the skip-gram model.
    def gen_batch(self, batch_size):
        if self.data_index + (batch_size // self.num_skips) * self.skip_window > len(self.data):
            return None, None, None, None, None
        assert batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.zeros(shape=(batch_size, 2), dtype=np.int32)
        labels = np.zeros(shape=(batch_size, 2), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch[:, 0], batch[:, 1], labels[:, 0], labels[:, 1], None

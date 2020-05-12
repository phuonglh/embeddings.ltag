from __future__ import print_function, division
from gensim.models import Word2Vec
import codecs
import time
# define training data
filename = "/home/vu/billion/all.txt"
binname = "/home/vu/billion/gensim.bin"
outname = "/home/vu/billion/gensim.txt"

#filename = "data/penn.text"
#binname = "data/gensim.bin"
#outname = "data/gensim.txt"

start = time.time()
with codecs.open(filename, "r", encoding="utf-8") as src:
    sentences = [x.lower().strip().split() for x in src.readlines()]
# train model

print(time.time() - start)
model = Word2Vec(sentences, size=100, min_count=10, workers=40, sg=1, iter=5)
print(time.time() - start)

# summarize the loaded model
# print(model)
# summarize vocabulary
words = list(model.wv.vocab)
with codecs.open(outname, "w", encoding="utf-8") as tgt:
    for word in words:
        tgt.write(word + "\t" + "\t".join(map(str, model[word])) + "\n")
# save model
model.save(binname)
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
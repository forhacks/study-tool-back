import numpy as np
import re
from gensim.models import Word2Vec

max_len = 25


def process_def(definition):
    if len(definition) == 1:
        return definition
    definition = re.sub("[^a-zA-Z\s]", " ", definition).split()
    definition = [a for a in definition if len(a) > 2]

    def_arr = np.array(definition)
    word_count = len(def_arr)

    stretch = ([max_len/word_count + 1] * (max_len % word_count))
    stretch.extend([max_len/word_count] * (word_count - max_len % word_count))
    def_arr = np.repeat(def_arr, stretch)
    return def_arr

print('-- READING DATA --')

# read in defs
with open("data/definitions.txt") as f:
    data = f.readlines()

# remove \n at end
data = [process_def(x.strip()) for x in data]

# reorganize into [[def 1, def 2, 1/0], [def 1, def 2, 1/0], ...]
data = np.reshape(data, (-1, 3)).T

x = data[:2].T
y = [int(x) for x in data[2:][0]]

model = Word2Vec.load("trained/w2v/trained.w2v")

x = [[model.wv[word] for word in a] for a in x]

np.save('data/x.npy', x)
np.save('data/y.npy', y)

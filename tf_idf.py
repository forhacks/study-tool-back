import glob

import re
from gensim.corpora import MmCorpus, Dictionary
from gensim.models.tfidfmodel import *
from gensim.models import TfidfModel

texts = []

for fname in glob.glob('data/raw/news.*'):
    print('Starting', fname)

    with open(fname, 'r') as f:
        data = re.sub("[^a-zA-Z\s.]", " ", f.read()).split()
        texts.append(data)

print(texts)
dictionary = Dictionary(texts)
corpus = MmCorpus('data/text.mm')

tfidf = TfidfModel(corpus)
tfidf.save("trained/tfidf.model")

import re

import numpy as np
from gensim.corpora import MmCorpus, Dictionary
from gensim.models import TfidfModel, Word2Vec

# loading
tfidf = TfidfModel.load("trained/tfidf.model")
corpus = MmCorpus("data/text.mm")
w2v = Word2Vec.load("trained/w2v/trained.w2v")
dictionary = Dictionary.load("data/dict.dict")

# varibles
articles = ["a", "an", "the"]
threshold = 0.9


def compare(def1, def2):
    # process words and split into array
    def1 = re.sub("[^a-zA-Z\s]", " ", def1.lower()).split()
    def2 = re.sub("[^a-zA-Z\s]", " ", def2.lower()).split()
    def1 = [x for x in def1 if x not in articles]
    def2 = [x for x in def2 if x not in articles]

    def1_len = len(def1)
    def2_len = len(def2)

    # vectors of words in sentences
    def1v = w2v[def1]
    def2v = w2v[def2]

    # processing of definitions go here
    def1v = np.sum(def1v) / def1_len
    def2v = np.sum(def2v) / def2_len
    '''
    # tfidf weights of each word
    def1w = tfidf_values[dictionary.token2id[def1]]
    def2w = tfidf_values[dictionary.token2id[def2]]

    # take dot product of vector and weight
    def1v = np.dot(def1v, def1w)
    def2v = np.dot(def2v, def2w)
    '''
    # take cos distance
    difference = np.inner(def1v, def2v) / (np.linalg.norm(def1v) * np.linalg.norm(def2v))

    return difference


while True:
    try:
        str1 = input('1 > ')
        str2 = input('2 > ')
        print(compare(str1, str2))
    except Exception:
        pass

from gensim.models import TfidfModel, Word2Vec
from gensim.corpora import MmCorpus, Dictionary
import numpy as np
import re

tfidf = TfidfModel.load("trained/tfidf.model")
corpus = MmCorpus("data/text.mm")
dictionary = Dictionary.load("data/dict.dict")
w2v = Word2Vec.load("trained/w2v/trained.w2v")

def sentence2vec(def1, def2):
    def1 = re.sub("[^a-zA-Z\s]", " ", def1.lower()).split()
    def2 = re.sub("[^a-zA-Z\s]", " ", def2.lower()).split()
    def1 = [x for x in def1 if not x in articles]
    def2 = [x for x in def2 if not x in articles]

    # vectors of words in sentences
    def1v = w2v[def1]
    def2v = w2v[def2]

    # tfidf weights of each word
    def1w = tfidf_values[dictionary.token2id[def1]]
    def2w = tfidf_values[dictionary.token2id[def2]]

    # take dot product of vector and weight
    def1v = np.dot(def1v, def1w)
    def2v = np.dot(def2v, def2w)

    # take cos distance
    difference = np.inner(def1v, def2v)

    return difference
from gensim.models.tfidfmodel import *
from gensim.models.word2vec import *
import numpy as np
import re

# variables
tfidf = TfidfModel.load("trained/tfidf.model")
w2v = Word2Vec.load("trained/w2v/trained.w2v")
articles = ["a", "an", "the"]
threshold = 0.9


def compare(def1, def2):

    # process words and split into array
    def1 = re.sub("[^a-zA-Z\s]", " ", def1.lower()).split()
    def2 = re.sub("[^a-zA-Z\s]", " ", def2.lower()).split()
    def1 = [x for x in def1 if not articles.__contains__(x)]
    def2 = [x for x in def2 if not articles.__contains__(x)]

    # vectors of words in sentences
    def1v = w2v[def1]
    def2v = w2v[def2]

    # weights of each word
    def1w = tfidf[def1]
    def2w = tfidf[def2]

    # take dot product of vector and weight
    def1v = np.dot(def1v, def1w)
    def2v = np.dot(def2v, def2w)

    # take cos distance
    difference = np.inner(def1v, def2v)

    return difference

compare("This is a test", "Test this is")

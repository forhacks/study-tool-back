from gensim.models.tfidfmodel import *
from gensim.models.word2vec import *
import numpy as np

#varibles
tfidf = TfidfModel.load("trained/tfidf.model")
w2v = Word2Vec.load("trained/trained.w2v")

#input is two strings
def compare(def1, def2):
    def1 = def1.split(" ")
    def2 = def2.split(" ")

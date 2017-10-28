from gensim.corpora import MmCorpus
from gensim.models.tfidfmodel import *


corpus = MmCorpus('data/text.mm')

tfidf = TfidfModel(corpus)
tfidf.save("trained/tfidf.model")

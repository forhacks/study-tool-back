from gensim.models.tfidfmodel import *

tfidf = TfidfModel("data/text.txt")
tfidf.save("trained/tfidf.model")

import glob
import re

from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus

path = "data/raw/news.*"

texts = []

for fname in glob.glob(path)[:2]:
    print('Starting', fname)

    with open(fname, 'r') as f:
        data = re.sub("[^a-zA-Z\s.]", " ", f.read()).split()
        texts.append(data)

print(texts)

dictionary = Dictionary(texts)
dictionary.save("data/dict.dict")

corpus = [dictionary.doc2bow(text) for text in texts]
MmCorpus.serialize('data/text.mm', corpus)

tfidf_model = TfidfModel(corpus, id2word=dictionary)
tfidf_model.save("trained/tfidf.model")

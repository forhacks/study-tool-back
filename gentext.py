import glob
import re

path = "data/raw/news.*"

fout = open('data/text.txt', 'w')

for fname in glob.glob(path):
    print('Starting', fname)
    with open(fname, 'r') as f:
        data = f.read()
        text = re.sub("[^a-zA-Z\s.]", " ", data)
        text = " ".join(text.split())
        fout.write(text)
fout.close()

import glob
import re

path = "trained/raw/news.*"

fout = open('data/text.txt', 'w')

for fname in glob.glob(path):
    with open(fname, 'r') as f:
        data = f.read()
        text = re.sub("[^a-zA-Z\s.]", " ", data)
        text = " ".join(text.split())
        fout.write(text)
fout.close()

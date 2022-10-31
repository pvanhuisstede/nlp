# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_word_embeddings.ipynb.

# %% auto 0
__all__ = ['documents', 'model', 'target_word', 'selected_words', 'embeddings', 'mapped_embeddings', 'x', 'y', 'nlp', 'word2pos',
           'sizes', 'windows', 'df', 'Corpus', 'evaluate']

# %% ../00_word_embeddings.ipynb 4
import sys
import os
import csv
import spacy
import gensim
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE
import numpy as np

# %% ../00_word_embeddings.ipynb 5
class Corpus(object):
    def __init__(self, filename):
        self.filename = filename
        self.nlp = spacy.blank("en")

    def __iter__(self):
        with open(self.filename, "r") as i:
            reader = csv.reader(i, delimiter=',')
            for _, abstract in reader:
                tokens = [t.text.lower() for t in self.nlp(abstract)]
                yield tokens

documents = Corpus("/home/peter/data/nlp/arxiv.csv")
model = gensim.models.Word2Vec(documents, min_count=100, window=5, vector_size=100) # We use the default CBOW algorithm

# %% ../00_word_embeddings.ipynb 8
model.wv["nlp"]

# %% ../00_word_embeddings.ipynb 10
print(model.wv.similarity("nmt", "smt"))
print(model.wv.similarity("nmt", "ner"))

# %% ../00_word_embeddings.ipynb 12
model.wv.similar_by_word("bert", topn=10)

# %% ../00_word_embeddings.ipynb 14
model.wv.most_similar(positive=["transformer", "lstm"], negative=["bert"], topn=1)

# %% ../00_word_embeddings.ipynb 16
model.wv.most_similar(positive=["tree"], topn=10)

# %% ../00_word_embeddings.ipynb 18
model.wv.most_similar(positive=["tree"], negative=["syntax"], topn=10)

# %% ../00_word_embeddings.ipynb 20
print(model.wv.doesnt_match("lst cnn gru transformer svm".split()))

# %% ../00_word_embeddings.ipynb 22
target_word = "bert"
selected_words = [w[0] for w in model.wv.most_similar(positive=[target_word], topn=200)] + [target_word]
embeddings = [model.wv[w] for w in selected_words] + model.wv["bert"]

mapped_embeddings = TSNE(n_components=2, metric='cosine', init='pca').fit_transform(embeddings)

# %% ../00_word_embeddings.ipynb 23
plt.figure(figsize=(20,20))
x = mapped_embeddings[:,0]
y = mapped_embeddings[:,1]
plt.scatter(x, y)

for i, txt in enumerate(selected_words):
    plt.annotate(txt, (x[i], y[i]))

# %% ../00_word_embeddings.ipynb 25
nlp = spacy.load('en_core_web_sm')

word2pos = {}
for word in model.wv.key_to_index: # model call can be 
    word2pos[word] = nlp(word)[0].pos_

word2pos["translation"]

# %% ../00_word_embeddings.ipynb 26
def evaluate(model, word2pos):
    same = 0
    for word in model.wv.key_to_index:
        most_similar = model.wv.similar_by_word(word, topn=1)[0][0]
        if word2pos[most_similar] == word2pos[word]:
            same = same + 1
    return same/len(model.wv.key_to_index)

evaluate(model, word2pos)

# %% ../00_word_embeddings.ipynb 28
sizes = [100, 200, 300]
windows = [2,3,5,10]

df = pd.DataFrame(index=windows, columns=sizes)

for size in sizes:
    for window in windows:
        print("Size:", size, "Window:", window)
        model = gensim.models.Word2Vec(documents, min_count=100, window=window, vector_size=size)
        acc = evaluate(model, word2pos)
        df[size][window] = acc
        
df

# %% ../00_word_embeddings.ipynb 30
df.plot()

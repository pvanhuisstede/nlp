# AUTOGENERATED! DO NOT EDIT! File to edit: ../02_discovering_topics.ipynb.

# %% auto 0
__all__ = ['f', 'df', 'question', 'nlp', 'texts', 'spacy_docs', 'docs', 'bigram', 'dictionary', 'corpus', 'model']

# %% ../02_discovering_topics.ipynb 2
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import re
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import warnings

# %% ../02_discovering_topics.ipynb 6
f = "/home/peter/Documents/data/nlp/la-transition-ecologique.csv"
df = pd.read_csv(f, low_memory=False)

# %% ../02_discovering_topics.ipynb 7
df.columns

# %% ../02_discovering_topics.ipynb 9
question = "QUXVlc3Rpb246MTU5 - Y a-t-il d'autres points sur la transition écologique sur lesquels vous souhaiteriez vous exprimer ?"
df[question].head(10)

# %% ../02_discovering_topics.ipynb 11
nlp = spacy.load('fr_core_news_sm')

# %% ../02_discovering_topics.ipynb 13
texts = df[df[question].notnull()][question]

# %% ../02_discovering_topics.ipynb 15
spacy_docs = list(nlp.pipe(texts))

# %% ../02_discovering_topics.ipynb 17
docs = [[t.lemma_.lower() for t in doc if len(t.orth_) > 3 and not t.is_stop] for doc in spacy_docs]

# %% ../02_discovering_topics.ipynb 20
bigram = Phrases(docs, min_count=10)

for idx in range(len(docs)):
  for token in bigram[docs[idx]]:
    if '_' in token: # bigrams can be picked out by using the '_' that joins the individual words
      docs[idx].append(token) # appended to the end, but topic modelling is BoW, so order is not important!

# %% ../02_discovering_topics.ipynb 22
docs[4]

# %% ../02_discovering_topics.ipynb 24
dictionary = Dictionary(docs)
print(f"Number of unique words in original documents: {len(dictionary)}")

dictionary.filter_extremes(no_below=3, no_above=0.25)
print(f"Number of unique words after removing rare and common words: {len(dictionary)}")

# Let's look at an example document:
print(f"Example representation of document 5: {dictionary.doc2bow(docs[5])}")

# %% ../02_discovering_topics.ipynb 26
corpus = [dictionary.doc2bow(doc) for doc in docs]
corpus[5]

# %% ../02_discovering_topics.ipynb 28
model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, chunksize=1000, passes=5, random_state=1)

# %% ../02_discovering_topics.ipynb 31
for (topic, words) in model.print_topics():
  print(topic + 1, ":", words)

# %% ../02_discovering_topics.ipynb 33
pyLDAvis.enable_notebook()
warnings.filterwarnings("ignore", category=DeprecationWarning)

pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)

# %% ../02_discovering_topics.ipynb 35
for (text, doc) in zip(texts[:10], docs[:10]):
    print(text)
    print([(topic+1, prob) for (topic, prob) in model[dictionary.doc2bow(doc)] if prob > 0.1])

#!/usr/bin/env python
# coding: utf-8

# # An introduction to word embeddings
# (follows: https://github.com/nlptown/nlp-notebooks/blob/master/An%20Introduction%20to%20Word%20Embeddings.ipynb)

# ## Training word embeddings
# 
# We use Gensim. We use the abstracts of all arXiv papers in the category cs.CL (CL: Computation and Language) published before mid-April 2021 (c. 25_000 documents). We tokenize the abstracts with spaCy.

# In[4]:


#| export
import sys
import os
import csv
import spacy

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

documents = Corpus("data/arxiv.csv")


# Using Gensim we can set a number of parameters for training:
# 
# - min_count: the minimum frequency of words in our corpus
# - window: number of words to the left and right to make up the context that word2vec will take into account
# - vector_size: the dimensionality of the word vectors; usually between 100 and 1_000
# - sg: One can choose fro 2 algorithms to train word2vec: Skip-gram (sg) tries to predict the context on the basis of the target word; CBOW tries to find the target on the basis of the context. Default is sg=0, hence: default is CBOW.

# In[5]:


#| export
import gensim

model = gensim.models.Word2Vec(documents, min_count=100, window=5, vector_size=100)


# ## Using word embeddings
# 
# With the model trained, we can access the word embedding via the **wv attribute** on model using the token as a key. For example the embedding for "nlp" is:

# In[6]:


#| export
model.wv["nlp"]


# **Find the similarity between two words.** We use the cosine between two word embeddings, so we use a ranges between -1 and +1. The higher the cosine, the more similar two words are.

# In[7]:


#| export
print(model.wv.similarity("nmt", "smt"))
print(model.wv.similarity("nmt", "ner"))


# **Find words that are most similar to target words** we line up words via the embeddings: semantically related, other types of pre-tained models, related general models, and generally related words:

# In[8]:


#| export
model.wv.similar_by_word("bert", topn=10)


# **Look for words that are similar to something, but dissimilar to something else** with this we can look for a kind of **analogies**:

# In[9]:


#| export
model.wv.most_similar(positive=["transformer", "lstm"], negative=["bert"], topn=1)


# So a related transformer to lstm is rnn, just like bert is a particular type of transformer; really powerful.
# 
# We can also zoom in on **one of the meanings of ambiguous words**. In NLP **tree** has a very specific meaning, is nearest neighbours being: constituency, parse, dependency, and syntax:

# In[10]:


#| export
model.wv.most_similar(positive=["tree"], topn=10)


# If we add **syntax** as a negative input to the query, we see that the ordinary meaning of tree kicks in: Now forest is one of the nearest neighbours.

# In[11]:


#! export
model.wv.most_similar(positive=["tree"], negative=["syntax"], topn=10)


# **Throw a list of words at the model** and filter out the odd one (here svm is the only non-neural model):

# In[12]:


#| export
print(model.wv.doesnt_match("lst cnn gru transformer svm".split()))


# ## Plotting embeddings
# 
# About visualizing embeddings. We need to reduce our 100-dimensions space to 2-dimensions. We can use t-SNE method: map similar data to nearby points and dissimilar data to faraway points in low dimensional space.
# 
# t-SNE is present in Scikit-learn. One has to specify two parameters: **n_components** (number of dimensions) and **metric** (similarity metric, here: cosine).
# 
# In order NOT to overcrowd the image we use a subset of embeddings of 200 most similar words based on a **target word**.

# In[13]:


#| export
#%matplotlib inline

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE

target_word = "bert"
selected_words = [w[0] for w in model.wv.most_similar(positive=[target_word], topn=200)] + [target_word]
embeddings = [model.wv[w] for w in selected_words] + model.wv["bert"]

mapped_embeddings = TSNE(n_components=2, metric='cosine', init='pca').fit_transform(embeddings)


# In[14]:


#| export
plt.figure(figsize=(20,20))
x = mapped_embeddings[:,0]
y = mapped_embeddings[:,1]
plt.scatter(x, y)

for i, txt in enumerate(selected_words):
    plt.annotate(txt, (x[i], y[i]))


# ## Exploring hyperparameters
# 
# What is the quality of the embeddings? Should embeddings capture syntax or semantical relations. Semantic similarity or topical relations?
# 
# One way of monitoring the quality is to check nearest neighbours: Are they two nouns, two verbs?

# In[16]:


#| export
import spacy

nlp = spacy.load('en_core_web_sm')

word2pos = {}
for word in model.wv.key_to_index: # model call can be 
    word2pos[word] = nlp(word)[0].pos_

word2pos["translation"]


# In[17]:


#| export
import numpy as np

def evaluate(model, word2pos):
    same = 0
    for word in model.wv.key_to_index:
        most_similar = model.wv.similar_by_word(word, topn=1)[0][0]
        if word2pos[most_similar] == word2pos[word]:
            same = same + 1
    return same/len(model.wv.key_to_index)

evaluate(model, word2pos)


# Now we want to change some of the settings we used above:
# 
# - embedding size (dimensions of the trained embeddings): 100, 200, 300
# - context window: 2, 5, 10
# 
# We will use a Pandas dataframe to keep track of the different scores (but this will take time: We train 9 models!!!):

# In[18]:


#| export
sizes = [100, 200, 300]
windows = [2,5,10]

df = pd.DataFrame(index=windows, columns=sizes)

for size in sizes:
    for window in windows:
        print("Size:", size, "Window:", window)
        model = gensim.models.Word2Vec(documents, min_count=100, window=window, vector_size=size)
        acc = evaluate(model, word2pos)
        df[size][window] = acc
        
df


# Results are close:
# 
# 1. Smaller contexts seem to yield better results. Which makes sense because we work with the syntax - nearer words often produce more information.
# 2. Higher dimension word embeddings not always work better than lower dimension. Here we have a relatively small corpus, not enough data for such higher dimensions.
# 
# Let's visualize our findings:

# In[19]:


df.plot()


# ## Conclusions
# 
# Word embeddings allow us to model the usage and meaning of a word, and discover words that behave in a similar way.
# 
# We move from raw strings -> vector space: word embeddings which allows us to work with words that have a similar meaning and discover new patterns.

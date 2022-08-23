# AUTOGENERATED! DO NOT EDIT! File to edit: ../04_ner_crf.ipynb.

# %% auto 0
__all__ = ['train_sents', 'dev_sents', 'test_sents', 'word2cluster', 'X_train', 'y_train', 'X_dev', 'y_dev', 'X_test', 'y_test',
           'crf', 'OUTPUT_PATH', 'OUTPUT_FILE', 'y_pred', 'example_sent', 'labels', 'sorted_labels', 'params_space',
           'f1_scorer', 'rs', 'best_crf', 'read_clusters', 'word2features', 'sent2features', 'sent2labels',
           'sent2tokens']

# %% ../04_ner_crf.ipynb 5
#%pip install git+https://github.com/MeMartijn/updated-sklearn-crfsuite.git\#egg=sklearn_crfsuite

# %% ../04_ner_crf.ipynb 7
import nltk
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics

# %% ../04_ner_crf.ipynb 8
#nltk.download('conll2002') # Just run this line once
train_sents = list(nltk.corpus.conll2002.iob_sents('ned.train'))
dev_sents = list(nltk.corpus.conll2002.iob_sents('ned.testa'))
test_sents = list(nltk.corpus.conll2002.iob_sents('ned.testb'))

# %% ../04_ner_crf.ipynb 10
train_sents[0]

# %% ../04_ner_crf.ipynb 13
# Making use of the Wikipedia word embeddings
def read_clusters(cluster_file):
  word2cluster = {}
  with open(cluster_file) as i:
    for line in i:
      word, cluster = line.strip().split('\t')
      word2cluster[word] = cluster
  return word2cluster

# Using features of the words AND looking at the context of a token (neigbours +/- 2)
def word2features(sent, i, word2cluster):
  word = sent[i][0]
  postag = sent[i][1]
  features = [
    'bias',
    'word.lower=' + word.lower(),
    'word[-3]=' + word[-3:], # looking at the last 3 chars of the token
    'word[-2]=' + word[-2:], # looking at the last 2 chars of the token
    'word.isupper=%s' % word.isupper(),
    'word.istitle=%s' % word.istitle(),
    'word.isdigit=%s' % word.isdigit(),
    'word.cluster=%s' % word2cluster[word.lower()] if word.lower() in word2cluster else '0'
    'postag=' + postag
  ]
  # Look at the first neighbour token to the left
  if i > 0:
    word1 = sent[i-1][0]
    postag1 = sent[i-1][1]
    features.extend([
      '-1:word.lower=' + word1.lower(),
      '-1:word.istitle=%s' % word1.istitle(),
      '-1:word.isupper=%s' % word1.isupper(),
      '-1:postag=' + postag1
    ])
  else:
    features.append('BOS')
  # Look at the second neighbour to the left
  if i > 1: 
    word2 = sent[i-2][0]
    postag2 = sent[i-2][1]
    features.extend([
      '-2:word.lower=' + word2.lower(),
      '-2:word.istitle=%s' % word2.istitle(),
      '-2:word.isupper=%s' % word2.isupper(),
      '-2:postag=' + postag2
    ])
  # look at the first neigbour to the right
  if i < len(sent)-1:
    word1 = sent[i+1][0]
    postag1 = sent[+1][0]
    features.extend([
      '+1:word.lower=' + word1.lower(),
      '+1:word.istitle=%s' % word1.istitle(),
      "+1:word.isupper=%s" % word1.isupper(),
      '+1:postag=' + postag1
    ])
  else:
    features.append('EOS')
  # Look at the second neighbour to the right
  if i < len(sent)-2:
    word2 = sent[i+2][0]
    postag2 = sent[+2][0]
    features.extend([
      '+2:word.lower=' + word2.lower(),
      '+2:word.istitle=%s' % word2.istitle(),
      "+2:word.isupper=%s" % word2.isupper(),
      '+2:postag=' + postag2
    ])
  return features

# Now we define the functions to do all the work
def sent2features(sent, word2cluster):
  return [word2features(sent, i, word2cluster) for i in range(len(sent))]

def sent2labels(sent):
  return [label for token, postag, label in sent]

def sent2tokens(sent):
  return [token for token, postag, label in sent]

word2cluster = read_clusters('/home/peter/Documents/data/nlp/clusters_nl.tsv')

# %% ../04_ner_crf.ipynb 15
train_sents[0][0]

# %% ../04_ner_crf.ipynb 16
sent2features(train_sents[0], word2cluster)[0]

# %% ../04_ner_crf.ipynb 18
X_train = [sent2features(s, word2cluster) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_dev = [sent2features(s, word2cluster) for s in dev_sents]
y_dev = [sent2labels(s) for s in dev_sents]

X_test = [sent2features(s, word2cluster) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


# %% ../04_ner_crf.ipynb 20
crf = crfsuite.CRF(
    verbose='true',
    algorithm='lbfgs',
    max_iterations=100
)

crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)

# %% ../04_ner_crf.ipynb 22
import joblib
import os

OUTPUT_PATH = '/home/peter/Documents/data/nlp/models'
OUTPUT_FILE = 'crf_model'

if not os.path.exists(OUTPUT_PATH):
  os.mkdir(OUTPUT_PATH)

joblib.dump(crf, os.path.join(OUTPUT_PATH, OUTPUT_FILE))

# %% ../04_ner_crf.ipynb 24
crf = joblib.load(os.path.join(OUTPUT_PATH, OUTPUT_FILE))
y_pred = crf.predict(X_test)

example_sent = test_sents[0]
print("Sentence:", ' '.join(sent2tokens(example_sent)))
print("Predicted:", ' '.join(crf.predict([sent2features(example_sent, word2cluster)])[0]))
print("Correct:", ' '.join(sent2labels(example_sent)))

# %% ../04_ner_crf.ipynb 26
labels = list(crf.classes_)
labels.remove('O')
y_pred = crf.predict(X_test)
sorted_labels = sorted(
  labels,
  key=lambda name: (name[1:], name[0])
)
# The following code only runs with the updated metrics.py module in `sklearn_crfsuite` library.
# Here: pip install git+https://github.com/MeMartijn/updated-sklearn-crfsuite.git\#egg=sklearn_crfsuite
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels))

# %% ../04_ner_crf.ipynb 31
import scipy
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

crf = crfsuite.CRF(
  algorithm='lbfgs',
  max_iterations=100,
  all_possible_transitions=True,
  keep_tempfiles=True
)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)

# %% ../04_ner_crf.ipynb 33
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# %% ../04_ner_crf.ipynb 34
best_crf = rs.best_estimator_
y_pred = best_crf.predict(X_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

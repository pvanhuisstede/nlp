# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_spacy_ner.ipynb.

# %% auto 0
__all__ = ['nlp', 'text', 'doc', 'entities', 'ner_lst', 'train_data', 'test', 'ner', 'pipe_exceptions', 'unaffected_pipes',
           'output_dir', 'nlp_updated', 'ruler', 'patterns', 'corpus', 'TRAIN_DATA', 'LABEL', 'optimizer', 'move_names',
           'other_pipes', 'test_text']

# %% ../03_spacy_ner.ipynb 3
from IPython.display import display_html
import tabulate
import spacy

nlp = spacy.load('en_core_web_sm')
text = "Alexander Boris de Pfeffel Johnson (born 19 June 1964) is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2019; he lead the Vote Leave campaign for Brexit . "

doc = nlp(text)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
display(display_html(tabulate.tabulate(entities, tablefmt='html')))

# %% ../03_spacy_ner.ipynb 6
ner_lst = nlp.pipe_labels['ner']
print(ner_lst)
for i in ner_lst:
  print(f"{i}: {spacy.explain(i)}\n")

# %% ../03_spacy_ner.ipynb 8
train_data = [
  ("Boris Johnson announced his pending resignation on 7 July 2022.", {"entities": [(0,13,"PERSON"), (36,47,"EVENT"), (51,62,"DATE")]}),
  ("He will remain as prime minister until a new party leader is elected.", {"entities": [(18,32,"NORP"), (45,57,"NORP"), (61,68,"EVENT")]}),
  ("He served as Secretary of State for Foreign and Commonwealth Affairs from 2016 to 2018.", {"entities": [(13,68,"NORP"), (74,86,"DATE")]}),
  ("Boris Johnson served as Mayor of London from 2008 to 2016.", {"entities": [(0,13,"PERSON"), (24,39,"NORP"), (45,57,"DATE")]}),
  ("He became a prominent figure in the successful Vote Leave campaign for Brexit in the 2016 European Union (EU) membership referendum.", {"entities": [(47,66,"EVENT"), (71,77,"EVENT"), (85,89,"DATE"), (90,104,"ORG"), (106,108,"ORG"), (121,131,"EVENT")]}),
]

# %% ../03_spacy_ner.ipynb 10
test = "Boris Johnson announced his pending resignation on 7 July 2022."
test[36:47]

# %% ../03_spacy_ner.ipynb 12
nlp.pipe_names

# %% ../03_spacy_ner.ipynb 14
ner = nlp.get_pipe('ner')

# %% ../03_spacy_ner.ipynb 16
for _, annotations in train_data:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# %% ../03_spacy_ner.ipynb 18
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# %% ../03_spacy_ner.ipynb 20
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

with nlp.disable_pipes(*unaffected_pipes):
  for iteration in range(10):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
      texts, annotations = zip(*batch)
      # new way of updating nlp NOT using nlp.update() anymore
      example = []
      # update the model with iterating each text
      for i in range(len(texts)):
        doc = nlp.make_doc(texts[i])
        example.append(Example.from_dict(doc, annotations[i]))

      nlp.update(example, drop=0.5, losses=losses)

print(losses)

# %% ../03_spacy_ner.ipynb 22
text = "Alexander Boris de Pfeffel Johnson (born 19 June 1964) is a British politician who has served as Prime Minister of the United Kingdom and Leader of the Conservative Party since 2019; he lead the Vote Leave campaign for Brexit . "

doc = nlp(text)
entities = [(t.text, t.ent_iob_, t.ent_type_) for t in doc]
display(display_html(tabulate.tabulate(entities, tablefmt='html')))

# %% ../03_spacy_ner.ipynb 24
# save the model to a directory
output_dir = Path('content/')
nlp.to_disk(output_dir)
print(f"Saved model to: {output_dir}")

# %% ../03_spacy_ner.ipynb 25
# Load the saved model to predict
print(f"Loading from: {output_dir}")
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("Johnson is a controversial figure in British politics.")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# %% ../03_spacy_ner.ipynb 27
#Build upon the spaCy Small Model
nlp = spacy.blank("en")


#Sample text
text = "Treblinka is a small village in Poland. Wikipedia notes that Treblinka is not large."

#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

#List of Entities and Patterns
patterns = [
                {"label": "GPE", "pattern": "Treblinka"}
            ]

ruler.add_patterns(patterns)

doc = nlp(text)

#extract entities
for ent in doc.ents:
    print (ent.text, ent.start_char, ent.end_char, ent.label_)

# %% ../03_spacy_ner.ipynb 29
#Import the requisite library
import spacy

#Build upon the spaCy Small Model
nlp = spacy.load("en_core_web_sm")

#Sample text
text = "Treblinka is a small village in Poland. Wikipedia notes that Treblinka is not large."

corpus = []

doc = nlp(text)
for sent in doc.sents:
    corpus.append(sent.text)

#Build upon the spaCy Small Model
nlp = spacy.blank("en")

#Create the EntityRuler
ruler = nlp.add_pipe("entity_ruler")

#List of Entities and Patterns
patterns = [
                {"label": "GPE", "pattern": "Treblinka"}
            ]

ruler.add_patterns(patterns)


TRAIN_DATA = []

#iterate over the corpus again
for sentence in corpus:
    doc = nlp(sentence)
    
    #remember, entities needs to be a dictionary in index 1 of the list, so it needs to be an empty list
    entities = []
    
    #extract entities
    for ent in doc.ents:

        #appending to entities in the correct format
        entities.append([ent.start_char, ent.end_char, ent.label_])
        
    TRAIN_DATA.append([sentence, {"entities": entities}])

print (TRAIN_DATA)

# %% ../03_spacy_ner.ipynb 31
# Get the `ner` component of the pipeline
nlp = spacy.load('en_core_web_sm')
ner = nlp.get_pipe('ner')

# %% ../03_spacy_ner.ipynb 32
# Add the new label
LABEL = "FOOD"

# Training examples in the required format
TRAIN_DATA =[ ("Pizza is a common fast food.", {"entities": [(0, 5, "FOOD")]}),
              ("Pasta is an italian recipe", {"entities": [(0, 5, "FOOD")]}),
              ("China's noodles are very famous", {"entities": [(8,14, "FOOD")]}),
              ("Shrimps are famous in China too", {"entities": [(0,7, "FOOD")]}),
              ("Lasagna is another classic of Italy", {"entities": [(0,7, "FOOD")]}),
              ("Sushi is extemely famous and expensive Japanese dish", {"entities": [(0,5, "FOOD")]}),
              ("Unagi is a famous seafood of Japan", {"entities": [(0,5, "FOOD")]}),
              ("Tempura , Soba are other famous dishes of Japan", {"entities": [(0,7, "FOOD")]}),
              ("Udon is a healthy type of noodles", {"entities": [(0,4, "ORG")]}),
              ("Chocolate soufflé is extremely famous french cuisine", {"entities": [(0,17, "FOOD")]}),
              ("Flamiche is french pastry", {"entities": [(0,8, "FOOD")]}),
              ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
              ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
              ("Frenchfries are considered too oily", {"entities": [(0,11, "FOOD")]})
           ]

# %% ../03_spacy_ner.ipynb 34
# Add the new label to ner
ner.add_label(LABEL)

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# List of pipes you want to train
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

# List of pipes which should remain unaffected in training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# %% ../03_spacy_ner.ipynb 35
# Importing requirements
from spacy.util import minibatch, compounding
import random

# Begin training by disabling other pipeline components

with nlp.disable_pipes(*other_pipes):
  sizes = compounding(1.0, 4.0, 1.001)
  for iteration in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=sizes)
    for batch in batches:
      texts, annotations = zip(*batch)
      # new way of updating nlp NOT using nlp.update() anymore
      example = []
      # update the model with iterating each text
      for i in range(len(texts)):
        doc = nlp.make_doc(texts[i])
        example.append(Example.from_dict(doc, annotations[i]))

      nlp.update(example, drop=0.5, losses=losses)

print(losses)

# %% ../03_spacy_ner.ipynb 37
test_text = "I ate Sushi yesterday. Maggi is a common fast food "
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
  print(ent)

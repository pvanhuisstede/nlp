# AUTOGENERATED! DO NOT EDIT! File to edit: ../01_pretrained_models.ipynb.

# %% auto 0
__all__ = ['en', 'text', 'doc_en', 'tokens', 'features', 'entities', 'syntax', 'nl', 'text_nl', 'doc_nl', 'info']

# %% ../01_pretrained_models.ipynb 2
import spacy
from IPython.display import display_html
import tabulate
import stanza
import spacy_stanza

# %% ../01_pretrained_models.ipynb 4
en = spacy.load('en_core_web_sm')

# %% ../01_pretrained_models.ipynb 5
text = ("Donald John Trump (born June 14, 1946) is the 45th and former president of "
        "the United States.  Before entering politics, he was a businessman and television personality.")

print(f"Type of text: {type(text)}")
print(f"Length of text: {len(text)}")

# %% ../01_pretrained_models.ipynb 7
doc_en = en(text)

# %% ../01_pretrained_models.ipynb 8
list(doc_en.sents)

# %% ../01_pretrained_models.ipynb 9
print(len(list(doc_en.sents)))

# %% ../01_pretrained_models.ipynb 10
tokens = [[t] for t in doc_en]
display(display_html(tabulate.tabulate(tokens, tablefmt='html')))

# %% ../01_pretrained_models.ipynb 12
features = [[t.orth_, t.lemma_, t.pos_, t.tag_] for t in doc_en]
display(display_html(tabulate.tabulate(features, tablefmt='html')))

# %% ../01_pretrained_models.ipynb 14
entities = [(t.orth_, t.ent_iob_, t.ent_type_) for t in doc_en]
display(display_html(tabulate.tabulate(entities, tablefmt='html')))

# %% ../01_pretrained_models.ipynb 16
print([(ent.text, ent.label_) for ent in doc_en.ents])

# %% ../01_pretrained_models.ipynb 18
syntax = [[token.text, token.dep_, token.head.text] for token in doc_en]

# %% ../01_pretrained_models.ipynb 23
nl = spacy.load('nl_core_news_sm')
text_nl = ("Mark Rutte is minister-president van Nederland." "Hij is van de VVD en heeft een slecht geheugen.")

# %% ../01_pretrained_models.ipynb 24
doc_nl = nl(text_nl)

# %% ../01_pretrained_models.ipynb 26
info = [(t.lemma_, t.pos_, t.tag_, t.ent_iob_, t.ent_type_) for t in doc_nl]
display(display_html(tabulate.tabulate(info, tablefmt='html')))

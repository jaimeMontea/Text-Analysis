'''The following code performs a tSNE (t-distributed stochastic neighbor embedding)
   to compare the relative similarity between some books available in the nltk library.
   The tSNE is a statistical method for visualizing high-dimensional data by giving each
   datapoint a location in a two or three-dimensional map.'''

import os
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
nltk.download('punkt')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

path = 'The Project Gutenberg'

for subdir, dirs, files in os.walk(path):
    token_dict = dict()
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        try:
            text = shakes.read()
        except UnicodeDecodeError:
            print('Not able to process {}.'.format(file))
            continue
        lowers = text.lower()
        no_punctuation = lowers.translate(string.punctuation)
        token_dict[file] = no_punctuation
        
sw = set()
sw.update(tuple(nltk.corpus.stopwords.words('english')))

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=sw)
values = tfidf.fit_transform(token_dict.values())

tsne = TSNE()
values_embedded = tsne.fit_transform(values)
palette = sns.color_palette("bright", len(token_dict.keys()))
sns.scatterplot(values_embedded[:,0], values_embedded[:,1], hue=token_dict.keys(), 
                legend='full', palette=palette)

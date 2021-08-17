'''Example of a text analysis from the gutenberg data (free online books) available in the nltk library.
   Text is treated as a bag-of-words (words without order).
   More common words (such as articles, connectors, etc) are removed since they are not necesarily 
   relevant in a text analysis.
   The frequencies of words are displayed.'''


import nltk
from collections import defaultdict
import pandas as pd
from nltk.stem.snowball import FrenchStemmer

#Retrieving data

nltk.download('gutenberg')
nltk.download('stopwords')
books = gutenberg.fileids()

dict_books = defaultdict(list)
for book in books:
    data = gutenberg.raw(book)
    dict_books[book.split('-')[0]].append(data)

#Analyzing text
    
tokenizer = nltk.RegexpTokenizer(r'\w+')

def freq_stats_corpora():
    corpora = defaultdict(list)

    # Creation of token corpus per writter
    for writer in dict_books:
        for text in dict_books[writer]:
            corpora[writer] += tokenizer.tokenize(text)

    stats, freq = dict(), dict()

    for k, v in corpora.items():
        freq[k] = fq = nltk.FreqDist(v)
        stats[k] = {'total': len(v), 'unique': len(fq.keys())} 
        
    return (freq, stats, corpora)

# Retrieving statistiques
freq, stats, corpora = freq_stats_corpora()
df = pd.DataFrame.from_dict(stats, orient='index')

# Displaying frequencies
df.sort_values(by=['total'], ascending=False)\
  .plot(kind='bar', color="#f56900", title='Writers per number of words written')

#To remove the most used words for each text, a counter is going to be used
#to count the frequency of words in all texts. Then the words more used are going 
#to be removed from the corpus. 

freq_total = nltk.Counter()
for k, v in corpora.items():
    freq_total += freq[k]
    
stop_words = [a for a,b in freq_total.most_common(100)]
sw = set()
sw.update(stop_words)
sw.update(tuple(nltk.corpus.stopwords.words('english')))

#The next fucntion calculates the frequency of words without the stop_words or words more used.

def freq_stats_corpora2(lookup_table=[]):
    corpora = defaultdict(list)
    for writer in dict_books:
        for text in dict_books[writer]:
            tokens = tokenizer.tokenize(text)
            corpora[writer] += [w for w in tokens if not w in list(sw)]

    stats, freq = dict(), dict()

    for k, v in corpora.items():
        freq[k] = fq = nltk.FreqDist(v)
        stats[k] = {'total': len(v), 'unique': len(fq.keys())}
    return (freq, stats, corpora)

#The result is going to be displayed.
freq2, stats2, corpora2 = freq_stats_corpora2()
df2 = pd.DataFrame.from_dict(stats2, orient='index')      
df2.sort_values(by=['total'], ascending=False)\
   .plot(kind='bar', color="#f56900", title='Writers per number of words written without most common ones.')   

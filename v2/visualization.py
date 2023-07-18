import pandas as pd
from matplotlib import pyplot as plt
import preprocessing #local file with functions
import nltk, sklearn
import time

TIMES = 4 #how many plots are plotted
LIMIT = 25 #number of ngrams in one plot

def visualize_ngrams(ngrams: dict[str, int], limit: int = LIMIT) -> None:
    '''Plot an n-gram counter.'''
    for i in range(TIMES):
        sorted_ngrams = sorted(zip(ngrams.values(), ngrams.keys()), reverse=True)[limit*i:limit*(i+1)]
        features, counts = [item[1] for item in sorted_ngrams], [item[0] for item in sorted_ngrams]
        plt.figure(figsize=(20,5))
        plt.bar(range(len(features)), counts, tick_label=features)
        plt.xticks(rotation=45)
        plt.show()

def find_and_visualize(input_column: list[str], lemmatizer: nltk.stem.WordNetLemmatizer, 
                       permitted_PoS: list[str], max_length: int, n: int,
                       vectorizer: sklearn.feature_extraction.text.TfidfVectorizer) -> list[str]:
    '''Perform the entire process of processing and visualization.'''
    ngrams = preprocessing.find_counter(input_column, vectorizer, n) #get the counter
    visualize_ngrams(ngrams)
    sorted_ngrams = sorted(zip(ngrams.values(), ngrams.keys()), reverse=True)[:LIMIT*TIMES]
    return [item[1] for item in sorted_ngrams]
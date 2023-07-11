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

def find_and_visualize(df: pd.DataFrame, start_date: pd.Timestamp, input_column: str, date_column: str,
                       lemmatizer: nltk.stem.WordNetLemmatizer, permitted_PoS: list[str], max_length: int,
                       n: int, vectorizer: sklearn.feature_extraction.text.TfidfVectorizer) -> dict[str, int]:
    '''Perform the entire process of processing and visualization.'''
    df[date_column] = pd.to_datetime(df[date_column])
    df = preprocessing.fully_clean(df, start_date, input_column, lemmatizer, permitted_PoS, max_length) #clean text and limit to dates after given starting date
    ngrams = preprocessing.find_counter(df[input_column].tolist(), vectorizer, n) #get the counter
    visualize_ngrams(ngrams)
    sorted_ngrams = sorted(zip(ngrams.values(), ngrams.keys()), reverse=True)[:LIMIT*TIMES]
    return [item[1] for item in sorted_ngrams], df
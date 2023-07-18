import numpy as np
import pandas as pd
import collections, string, time, functools
import nltk, sklearn

IGNORE_STRING = 'IGNORE'
ENGLISH_WORDS = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('french')
GLOSSARY = ['kike', 'zog', 'george soros', 'rothschild']

#text cleaning
@functools.cache
def lemmatize(sentence: str, lemmatizer: nltk.stem.WordNetLemmatizer):
    '''Lemmatize a string. Defined as seperate function for cache functionality.'''
    return lemmatizer.lemmatize(sentence)

@functools.cache
def is_in_wordlist(word: str, wordlist: frozenset[str]) -> bool:
    '''Return whether a word is in a wordlist. This is in here for the cache functionality.'''
    return word in wordlist

def delete_words(text: str, wordlist: frozenset[str]) -> str:
    '''Remove all of the words in a wordlist from a text.'''
    return ' '.join(word for word in text.split() if not is_in_wordlist(word, wordlist))

def clean_text(text: str, keep_punctuation: bool = False) -> str:
    '''Clean a text, removing HTTP, digits, uppercase, digits, and punctuation.'''
    ending_index = len(text) if 'http' not in text else text.index('http')
    text = text[:ending_index]
    for number in range(10):
        text = text.replace(str(number), '')
    if not keep_punctuation:
        for punct in string.punctuation:
            text = text.replace(punct, '')
    return text.lower()

def preprocess_text(text: str, lemmatizer: nltk.stem.WordNetLemmatizer, permitted_PoS: list[str], max_length: int) -> str:
    '''Preprocess a text, tokenizing and lematizing it, preparing for extraction.'''
    text = [item[0] for item in nltk.pos_tag(text.split()) if item[1] in permitted_PoS]
    text = [item for item in text if len(item) < max_length and item not in ENGLISH_WORDS]
    return ' '.join(lemmatize(item, lemmatizer) for item in text)

def keep_words(text: str, words_to_keep: frozenset[str]) -> str:
    '''Keep only the words in the text given by `words_to_keep`'''
    return ' '.join(word for word in text.split() if is_in_wordlist(word, words_to_keep))

#dataframe wrangling
def limit_dates(df: pd.DataFrame, start_date: pd.Timestamp):
    '''Limit dates to only those after or on the start date.'''
    return df[df['Date'] >= start_date]

def fully_clean(df: pd.DataFrame, start_date: pd.Timestamp, input_column: str, lemmatizer: nltk.stem.WordNetLemmatizer, 
                permitted_PoS: list[str], max_length: int) -> pd.DataFrame:
    '''Perform all cleaning functions on a df.'''
    df = limit_dates(df, start_date)
    df[input_column] = df[input_column].apply(clean_text)
    df[input_column] = df[input_column].apply(lambda x: preprocess_text(x, lemmatizer, permitted_PoS, max_length))
    full_text = ' '.join(df[input_column]).split()
    text_counter = collections.Counter(full_text)
    to_delete_words = frozenset(word for word in text_counter if text_counter[word] <= 2) #delete words that only appear once or twice
    df[input_column] = df[input_column].apply(lambda x: delete_words(x, to_delete_words))
    return df

#ngram discovery
def max_tfidf(text: list[str], vectorizer: sklearn.feature_extraction.text.TfidfVectorizer) -> np.ndarray:
    '''Vectorize a text using TF-IDF, return maximum score across all documents.'''
    return np.max(vectorizer.fit_transform(text), axis=0).todense()

def tfidf_limit(tfidf_scores: np.ndarray, features: np.ndarray, threshold: float) -> list[str]:
    '''Limit the features in a wordlist to only those above a particular TF-IDF score `threshold`.'''
    return [word for (i, word) in enumerate(features) if tfidf_scores[0, i] >= threshold]

def get_all_ngrams(text: str, n: int) -> list[tuple]:
    '''Find all n-grams in a text.'''
    offsets = list(zip(*[text.split()[i:-i-n+1] for i in range(n)]))
    return [item for item in offsets if IGNORE_STRING not in item]

def find_ngram_frequency(ngrams: list[tuple]) -> dict[str, int]:
    '''Discover the n-grams in a text, returning a counter of those bigrams' frequencies.'''
    return collections.Counter(' '.join(ngram) for ngram in ngrams)

def find_counter(text: list[str], vectorizer: sklearn.feature_extraction.text.TfidfVectorizer, n: int) -> dict[str, int]:
    '''Find the overall counter of n-grams given'''
    tfidf_scores = max_tfidf(text, vectorizer)
    ngrams_to_keep = frozenset(word for word in tfidf_limit(tfidf_scores, vectorizer.get_feature_names_out(), np.percentile(tfidf_scores[:, 0], 80)) if word not in GLOSSARY)
    possible_ngrams = get_all_ngrams(f' {IGNORE_STRING} '.join(text), n)
    ngrams = [ngram for ngram in possible_ngrams if is_in_wordlist(' '.join(ngram), ngrams_to_keep)]
    return find_ngram_frequency(ngrams)
import pandas as pd
import numpy as np
import gensim, nltk
import preprocessing #local file with functions

FILENAME = 'model.model'

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    '''Find the cosine similarity of two vectors.'''
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

def get_model(filename: str, data: list[list[str]]) -> gensim.models.Word2Vec:
    '''Retrieve a model from the disk, or make a new one and save.'''
    try:
        model = gensim.models.Word2Vec.load(filename)
        return model
    except Exception:
        model = gensim.models.Word2Vec(data, min_count=5, vector_size=100, window=5, workers=5, sg=1)
        model.save(filename)
        return model

def compute_similarities(model: gensim.models.Word2Vec, compare_terms: list[str], terms_to_compare: list[str]) -> np.ndarray:
    '''Perform the cosine similarity computation on a list of comparison terms with a larger list of terms to compare to them.'''
    return np.array([[cos_sim(model.wv[term], model.wv[to_compare]) for term in compare_terms] for to_compare in terms_to_compare])

def get_similarities(df: pd.DataFrame, start_date: pd.Timestamp, input_column: str, date_column: str, ngrams: list[str], 
                     compare_terms: list[str], lemmatizer: nltk.stem.WordNetLemmatizer, permitted_PoS: list[str], max_length: int) -> pd.DataFrame:
    '''Get the similarities of all terms within an ngram list to a list of comparison terms.'''
    phrases = gensim.models.phrases.Phrases(df[input_column], threshold=5, min_count=5, connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
    df[input_column] = df[input_column].apply(lambda x: phrases[x.split()])
    model = get_model(FILENAME, df[input_column].tolist())
    ngrams = [ngram.replace(' ', '_') for ngram in ngrams if ngram.replace(' ', '_') in model.wv]
    output_df = pd.DataFrame(data=compute_similarities(model, compare_terms, ngrams), index=ngrams, columns=compare_terms)
    output_df['Mean'] = output_df.mean(axis=1)
    return output_df
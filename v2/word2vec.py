import pandas as pd
import numpy as np
import gensim, nltk
import preprocessing #local file with functions

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    '''Find the cosine similarity of two vectors.'''
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

def get_model(filename: str, data: list[list[str]]) -> gensim.models.Word2Vec:
    '''Retrieve a model from the disk, or make a new one and save.'''
    try:
        model = gensim.models.Word2Vec.load(filename)
        return model
    except Exception:
        model = gensim.models.Word2Vec(data, min_count=1, vector_size=200, window=5, workers=5, sg = 1)
        model.save(filename)
        return model

def compute_similarities(model: gensim.models.Word2Vec, compare_terms: list[str], terms_to_compare: list[str]) -> np.ndarray:
    '''Perform the cosine similarity computation on a list of comparison terms with a larger list of terms to compare to them.'''
    return np.array([[cos_sim(model.wv[term], model.wv[to_compare]) for term in compare_terms] for to_compare in terms_to_compare])

def get_similarities(df: pd.DataFrame, start_date: pd.Timestamp, input_column: str, date_column: str, ngrams: list[str], 
                     compare_terms: list[str], lemmatizer: nltk.stem.WordNetLemmatizer, permitted_PoS: list[str], max_length: int, n: int) -> pd.DataFrame:
    '''Get the similarities of all terms within an ngram list to a list of comparison terms.'''
    df = df.copy()
    for word in compare_terms:
        df[input_column] = df[input_column].apply(lambda x: x.replace(word.replace('_', ' '), word.replace(' ', '_')))
    for ngram in ngrams:
        df[input_column] = df[input_column].apply(lambda x: x.replace(ngram.replace('_', ' '), ngram.replace(' ', '_')))
    df[input_column] = df[input_column].apply(lambda x: x if x is list else x.split())
    model = get_model(f'model-{n}.model', df[input_column].tolist())
    ngrams = [ngram.replace(' ', '_') for ngram in ngrams if ngram.replace(' ', '_') in model.wv]
    output_df = pd.DataFrame(data=compute_similarities(model, compare_terms, ngrams), index=ngrams, columns=compare_terms)
    output_df['Mean'] = output_df.mean(axis=1)
    return output_df
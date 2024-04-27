import ast
import json
import re
import math
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')

STOPWORDS = frozenset([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fifty', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once'
])

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)

def remove_stopword_tokens(tokens, stopwords=None):
    """Remove stopword tokens using list `stopwords`.

    Parameters
    ----------
    tokens : iterable of str
        Sequence of tokens.
    stopwords : iterable of str, optional
        Sequence of stopwords
        If None - using :const:`~gensim.parsing.preprocessing.STOPWORDS`

    Returns
    -------
    list of str
        List of tokens without `stopwords`.

    """
    if stopwords is None:
        stopwords = STOPWORDS
    return [token for token in tokens if token not in stopwords]


def remove_stopwords(s, stopwords=None):
    s = to_unicode(s)
    return " ".join(remove_stopword_tokens(s.split(), stopwords))

def preprocess(json_file):
    """converts the json to a pandas dataframe (returns). Averages the 0-100 score and 
    turns the list of reviews into a list of tokens (from all of the reviews)."""
    df = pd.read_json(json_file)
    df.drop("Release_year", inplace=True, axis=1)
    df["Score"] = df["Score"].apply(lambda lst: [float(i) for i in lst[1:-1].split(", ")])
    df["Score"] = df["Score"].apply(lambda lst: sum(lst)/len(lst))
    df["Review"] = df["Review"].apply(lambda reviews: nltk.tokenize.word_tokenize(remove_stopwords((re.sub(r'[^\w\s]', '', reviews)).lower())))
    df = df.reset_index(drop=False)
    df = df.rename(columns={'index': 'ID'})
    df['Platform'] = df['Platform'].apply(lambda x: ast.literal_eval(x))

    flattened = [item for sublist in df['Platform'] for item in sublist]
    # Get unique values
    unique_values = set(flattened)
    print(unique_values)
    return df

def reviewOutput(json_file, doc_id):
   """Returns the string output of a review"""
   with open(json_file, 'r') as user_file:
    parsed_json = json.load(user_file)

    item = parsed_json[doc_id]
    reviews = []
    if item['Review'].startswith("[\"") or item['Review'].startswith("['"):
        # Normalize quotes and parse as JSON
        reviews = json.loads(item['Review'].replace("'", "\""))
    
    if reviews:
        return reviews[0]

def token_inverted_index(df):
    """Takes in the dataframe created from the json and produces an inverted index. 
    returns a dictionary where Each token is a key and the indexes of games with
    reviews with that token are the values
    returns:
    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]"""

    temp = {}
    for _, row in df.iterrows():
        for token in row["Review"]:
            if token not in temp:
                temp[token] = {}
            if row["ID"] not in temp[token]:
                temp[token][row["ID"]]=0
            temp[token][row["ID"]]+=1
    
    inv_idx = {}
    for key, val in temp.items():
      inv_idx[key] = [(k, v) for k, v in val.items()]
    return inv_idx

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======
    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = {}
    for word in inv_idx.keys():
        doc_count = len(inv_idx[word])
        if doc_count >= min_df and (doc_count/n_docs) <= max_df_ratio:
            idf[word] = math.log((n_docs/(1+doc_count)), 2.0)
    return idf

def compute_doc_norms(inv_idx, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    norms = np.zeros((n_docs))
    for word, docs in inv_idx.items():
      if word in idf:
        for (doc_index, freq) in docs:
          norms[doc_index]+=((freq*idf[word])**2)

    return np.sqrt(norms)

def accumulate_dot_scores(query_word_counts, index: dict, idf: dict):
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    
    doc_scores = {}
    for qword, qword_freq in query_word_counts.items():
      if qword in idf:
        for dID, d_Freq in index[qword]:
          if dID not in doc_scores:
            doc_scores[dID] = 0
          doc_scores[dID] += (qword_freq*d_Freq)*(idf[qword]**2)

    return doc_scores

def index_search(
    df,
    query: str,
    index: dict,
    idf,
    doc_norms,
    console,
    score_func=accumulate_dot_scores,
):
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.
    index: an inverted index as above
    idf: idf values precomputed as above
    doc_norms: document norms as computed above
    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)
    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    """

    #convert query into dictionary of word:freq
    q_tokens = nltk.tokenize.word_tokenize(remove_stopwords((re.sub(r'[^\w\s]', '', query)).lower()))
    #convert tokens to dict
    query_word_counts = {}
    for token in q_tokens:
        if token not in query_word_counts:
            query_word_counts[token]=0
        query_word_counts[token]+=1

    #compute norm of query
    qnorm=0
    for word, freq in query_word_counts.items():
      if word in idf:
        qnorm+=((freq*idf[word])**2)
    qnorm=np.sqrt(qnorm)

    result = []
    for doc, score in score_func(query_word_counts, index, idf).items():
        if console == "any":
            cossim=score/(qnorm * doc_norms[int(doc)])
            result.append((cossim, doc))
        elif console in df[doc]["Platform"] :
            cossim=score/(qnorm * doc_norms[int(doc)])
            result.append((cossim, doc))
    
    return sorted(result, reverse=True, key=lambda x:x[0])

def apply_svd_to_documents(df):
    """Apply Singular Value Decomposition (SVD) to documents in the dataframe"""
    # Convert reviews to strings
    df['Review_str'] = df['Review'].apply(lambda x: ' '.join(x))
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Review_str'])
    
    # Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_result = svd.fit_transform(X)
    
    # Append SVD components to the dataframe
    for i in range(svd_result.shape[1]):
        df[f'SVD_{i+1}'] = svd_result[:, i]
    
    return df

# Assuming this function will be called when loading the JSON
def process_data(json_file):
    """Process data from the JSON file"""
    df = preprocess(json_file)
    df = apply_svd_to_documents(df)
    inv_idx = token_inverted_index(df)
    idf = compute_idf(inv_idx, len(df))
    norms = compute_doc_norms(inv_idx, idf, len(df))
    return df, inv_idx, idf, norms

# Assuming this function will be called when searching
def search_query(query, df, inv_idx, idf, norms):
    """Search for the query in the preprocessed data"""
    sorted_matches = index_search(query, inv_idx, idf, norms)
    final_list = []
    for sim, docID in sorted_matches[:10]:
        game_data = df.loc[df["ID"] == int(docID)]
        game_data.drop("Review", axis=1, inplace=True)
        game_data["Similarity"] = sim
        final_list.append(game_data.iloc[0].to_dict())
    if len(final_list) == 0:
        final_list.append({"Game" : "No results. Please try a different query.", "Score":0})
    return json.dumps(final_list)


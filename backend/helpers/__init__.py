import ast
import json
import re
import math
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
# from gensim.parsing.preprocessing import remove_stopwords

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
    return df

def filter(df, genre, platform_code):
    """returns a new dataframe with only rows that Genre matches param genre and
    Platform matches platform_code."""
    return df[df["Genre"]==genre & df["Platform"]==platform_code]

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
    query: str,
    index: dict,
    idf,
    doc_norms,
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

      cossim=score/(qnorm * doc_norms[int(doc)])
      result.append((cossim, doc))
    
    return sorted(result, reverse=True, key=lambda x:x[0])
        

#____________________________________________________________________________#



    # query_tokens = set(tokenize(query.lower()))
    # similarity_scores = {}

    # for game, reviews in game_reviews_dict.items():

    #     review_scores = []
    #     for review in reviews:
    #         review_tokens = set(tokenize(review.lower()))
    #         score = jaccard_similarity(review_tokens, query_tokens)
    #         review_scores.append(score)


    #     average_score = np.max(review_scores)


    #     similarity_scores[game] = average_score


    # sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # return sorted_similarity_scores


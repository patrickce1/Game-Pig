import ast
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def tokenize(text):
    # Starting with a simple tokenization pattern
    # This pattern does not cover all Treebank tokenizer rules but gives a basic idea
    token_pattern = r'''(?x)          # Set flag to allow verbose regexps
        (?:[A-Z]\.)+                  # Abbreviations, e.g., U.S.A.
        | \w+(?:-\w+)*                # Words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?          # Currency and percentages, e.g., $12.40, 82%
        | \.\.\.                      # Ellipsis
        | [][.,;"'?():_`-]            # These are separate tokens; includes ], [
    '''

    # Using regular expression to split the text into tokens
    tokens = re.findall(token_pattern, text)

    return tokens

def dict_creator(data):
  game_reviews_dict = {}
  for item in data:
      game_name = item['Game']
      reviews_str = item['Review']
      reviews = ast.literal_eval(reviews_str)
      tokenized_reviews = [tokenize(review) for review in reviews]
      
      game_reviews_dict[game_name] = tokenized_reviews
  return game_reviews_dict



def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

def compute_similarity_score(game_reviews_dict, query):
    vectorizer = TfidfVectorizer()

    cosine_scores = {}

    for game, reviews in game_reviews_dict.items():
        if reviews:
            # Combine the reviews with the query for TF-IDF transformation
            all_texts = reviews + [query]

            # Transform texts to a TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Compute cosine similarity between the query and each review
            # The query is the last document in the matrix
            cos_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])

            # Calculate the average cosine similarity score for the game
            average_score = np.mean(cos_similarities)

            cosine_scores[game] = average_score

    # Sort the games by their average cosine similarity score in descending order
    sorted_cosine_scores = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_cosine_scores



    # query_tokens = tokenize(query.lower())
    # similarity_scores = {}

    # for game, reviews in game_reviews_dict.items():
    #     review_tokens = [token for review in reviews for token in review]
    #     similarity = jaccard_similarity(review_tokens, query_tokens)
    #     similarity_scores[game] = similarity
    #     sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # return sorted_similarity_scores




    # query_tokens = set(tokenize(query.lower()))
    # similarity_scores = {}

    # for game, reviews in game_reviews_dict.items():
    #     # Compute similarity for each review and collect the scores
    #     review_scores = []
    #     for review in reviews:
    #         review_tokens = set(tokenize(review.lower()))
    #         score = jaccard_similarity(review_tokens, query_tokens)
    #         review_scores.append(score)

    #     # Calculate the average similarity score for the game
    #     average_score = np.max(review_scores)


    #     similarity_scores[game] = average_score

    # # Sort the games by their average similarity score in descending order
    # sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # return sorted_similarity_scores


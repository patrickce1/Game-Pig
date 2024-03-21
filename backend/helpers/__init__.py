import ast
import json
import re
import numpy as np
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

def compute_similarity_with_query(game_reviews_dict, query):
    # query_tokens = tokenize(query.lower())
    # similarity_scores = {}

    # for game, reviews in game_reviews_dict.items():
    #     review_tokens = [token for review in reviews for token in review]
    #     similarity = jaccard_similarity(review_tokens, query_tokens)
    #     similarity_scores[game] = similarity
    #     sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # return sorted_similarity_scores
    query_tokens = set(tokenize(query.lower()))
    similarity_scores = {}

    for game, reviews in game_reviews_dict.items():
        # Compute similarity for each review and collect the scores
        review_scores = []
        for review in reviews:
            review_tokens = set(tokenize(review.lower()))
            score = jaccard_similarity(review_tokens, query_tokens)
            review_scores.append(score)

        # Calculate the average similarity score for the game
        average_score = np.max(review_scores)


        similarity_scores[game] = average_score

    # Sort the games by their average similarity score in descending order
    sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_similarity_scores


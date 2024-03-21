# import ast
import json
# from nltk import TreebankWordTokenizer



# tokenizer = TreebankWordTokenizer()

# def dict_creator(data):
#   game_reviews_dict = {}
#   for item in data:
#       game_name = item['Game']
#       reviews_str = item['Review']
#       reviews = ast.literal_eval(reviews_str)
#       tokenized_reviews = [tokenizer.tokenize(review) for review in reviews]
      
#       game_reviews_dict[game_name] = tokenized_reviews



# def jaccard_similarity(tokens1, tokens2):
#     set1 = set(tokens1)
#     set2 = set(tokens2)
#     return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

# def compute_similarity_with_query(game_reviews_dict, query):
#     query_tokens = tokenizer.tokenize(query.lower())
#     similarity_scores = {}

#     for game, reviews in game_reviews_dict.items():
#         review_tokens = [token for review in reviews for token in review]
#         similarity = jaccard_similarity(review_tokens, query_tokens)
#         similarity_scores[game] = similarity
#         sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

#     return sorted_similarity_scores

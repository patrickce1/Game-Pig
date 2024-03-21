import json
import ast
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from nltk import TreebankWordTokenizer
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
     data = json.load(file)
#     episodes_df = pd.DataFrame(data['episodes'])
#     reviews_df = pd.DataFrame(data['reviews'])
tokenizer = TreebankWordTokenizer()
game_reviews_dict = {}
for item in data:
    game_name = item['Game']
    reviews_str = item['Review']
    reviews = ast.literal_eval(reviews_str)
    tokenized_reviews = [tokenizer.tokenize(review) for review in reviews]
    
    game_reviews_dict[game_name] = tokenized_reviews

# Check a few entries from the dictionary
# for game, reviews in list(game_reviews_dict.items())[:1]:
#     print(game, reviews[:2])



def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

# Function to compute Jaccard similarity between game reviews and a query string
def compute_similarity_with_query(game_reviews_dict, query):
    query_tokens = tokenizer.tokenize(query.lower())
    similarity_scores = {}

    for game, reviews in game_reviews_dict.items():
        # Flatten the list of tokenized reviews into a single list of tokens
        review_tokens = [token for review in reviews for token in review]
        
        # Compute Jaccard similarity
        similarity = jaccard_similarity(review_tokens, query_tokens)
        similarity_scores[game] = similarity
        sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_similarity_scores

# # Example query string
# query = "action adventure"

# # Compute the similarity scores
# similarity_scores = compute_similarity_with_query(game_reviews_dict, query)

# # Display the similarity scores
# for game, score in similarity_scores[:5]:
#     print(f"{game}: {score}")




app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    # matches = []
    # merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    # matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    # matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json
    similarity_scores = compute_similarity_with_query(game_reviews_dict, query)
    scores_df = pd.DataFrame(list(similarity_scores.items()), columns=['Game', 'Score'])
    top_matches = scores_df.nlargest(3, 'Score')
    top_matches_json = top_matches_filtered.to_json(orient='records')
    return top_matches_json
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
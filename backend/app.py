import json
import re 
import ast
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from helpers.__init__ import *
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# # Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    df = preprocess(json_file_path)
    inv_idx = token_inverted_index(df)
    idf = compute_idf(inv_idx, len(df))
    norms = compute_doc_norms(inv_idx, idf, len(df))



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

    
    #similarity_scores = compute_similarity_score(game_reviews_dict, query)
    query = str(query)

    sorted_matches = index_search(query, inv_idx, idf, norms)
    final_list = []
    for sim, docID in sorted_matches[:10]:
        game_data = df.loc[df["ID"] == int(docID)]
        game_data.drop("Review", axis=1, inplace=True)
        game_data["Similarity"] = sim
        final_list.append(game_data.iloc[0].to_dict())
    return json.dumps(final_list)
          

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
import json
import re 
import ast
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from helpers.__init__ import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Set up NLTK
nltk.download('punkt')

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    df = preprocess(json_file_path)
    inv_idx = token_inverted_index(df)
    idf = compute_idf(inv_idx, len(df))
    norms = compute_doc_norms(inv_idx, idf, len(df))

# Define SVD parameters
n_components = 100  # Adjust as needed
random_state = 42  # For reproducibility

# Preprocess text for SVD
text_corpus = df['Review'].apply(lambda x: ' '.join(x))

# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(text_corpus)

# Apply SVD
svd = TruncatedSVD(n_components=n_components, random_state=random_state)
svd_matrix = svd.fit_transform(tfidf_matrix)

app = Flask(__name__)
CORS(app)

def calculate_combined_similarity(query, doc_id):
    query_representation = svd.transform(tfidf_vectorizer.transform([query]))[0]
    doc_representation = svd_matrix[doc_id]
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(query_representation, doc_representation) / (np.linalg.norm(query_representation) * np.linalg.norm(doc_representation))

    # Calculate SVD similarity
    svd_similarity = np.dot(query_representation, doc_representation) / (np.linalg.norm(query_representation) * np.linalg.norm(doc_representation))

    # Combine cosine similarity and SVD similarity
    combined_similarity = 0.6 * cosine_similarity + 0.3 * svd_similarity + .1 * df[df["ID"] == doc_id]["Score"]/100 # Adjust weights as needed

    return combined_similarity

def json_search(query, console):
    query = str(query)
    sorted_matches = index_search(query, inv_idx, idf, norms)
    final_list = []
    print(type[df["Platform"]])
    if console != "any":
        filtered = []
        for score, doc_id in sorted_matches:
            print (df[df["ID"]==doc_id]["Platform"].tolist())
            if console in ((df[df["ID"]==doc_id]["Platform"]).tolist())[0]:
                filtered.append((score,doc_id))
        sorted_matches = filtered

    for _, docID in sorted_matches[:10]:
        game_data = df.loc[df["ID"] == int(docID)]
        
        # Calculate combined similarity and include it in the sorting
        combined_similarity = calculate_combined_similarity(query, int(docID))
        game_data["Similarity"] = combined_similarity
        text = reviewOutput(json_file_path, docID)
        print(type(text))
        print(text)
        game_data["Review"] = text
        print(text[0])
        final_list.append(game_data.iloc[0].to_dict())

    # Sort final list based on combined similarity
    final_list.sort(reverse=True, key=lambda x: x["Similarity"])

    if len(final_list) == 0:
        final_list.append({"Game": "No results. Please try a different query.", "Similarity": 0})

    return json.dumps(final_list)

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/episodes")
def episodes_search():
    print(request.args)
    text = request.args.get("title")
    console = request.args.get("console")
    return json_search(text, console)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)

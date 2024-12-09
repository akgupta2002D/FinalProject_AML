# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Step 1: Load the dataset
data = pd.read_csv("tmdb-movies.csv")

# Step 2: Data Preprocessing
# Select relevant features for the recommendation system
selected_features = ['genres', 'overview', 'cast', 'director']

# Fill missing values with an empty string
for feature in selected_features:
    data[feature] = data[feature].fillna('')

# Combine all selected features into a single column
data['combined_features'] = data['genres'] + ' ' + data['overview'] + ' ' + data['cast'] + ' ' + data['director']

# Step 3: Feature Extraction
# Convert the combined features into numerical vectors using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Step 4: Calculate Cosine Similarity
# Compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Define a function for movie recommendations
def recommend_movies(movie_title, data=data, cosine_sim_matrix=cosine_sim_matrix):
    print("\n# Step 1: Take movie name")
    print(f"Input movie: '{movie_title}'")
    
    # Find the closest match for the movie title in the dataset
    all_titles = data['original_title'].tolist()
    print("\n# Step 2: Finding the close match with input")
    close_matches = difflib.get_close_matches(movie_title, all_titles, n=1)
    print(f"Close matches found: {close_matches}")
    
    if not close_matches:
        return "Movie not found."
    
    # Closest match and its index
    closest_title = close_matches[0]
    print("\n# Step 3: Closest match - the searched movie most of the time")
    print(f"Closest match: '{closest_title}'")
    
    movie_index = data[data['original_title'] == closest_title].index[0]
    print("\n# Step 4: Index of closest match")
    print(f"Index of the closest match: {movie_index}")
    
    # Get similarity scores for the movie
    print("\n# Step 5: Make a list with similarity score and index, of that movie")
    similarity_scores = list(enumerate(cosine_sim_matrix[movie_index]))
    print(f"Similarity scores (index, score): {similarity_scores[:5]} ...")  # Print first few scores for brevity
    
    # Sort movies based on similarity scores in descending order
    print("\n# Step 6: Sort to get most similar movies at first")
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    print(f"Sorted similarity scores (index, score): {sorted_movies[:5]} ...")  # Print first few sorted scores
    
    # Get the top 10 most similar movies (excluding the first one, which is the same movie)
    recommended_indices = [movie[0] for movie in sorted_movies[1:11]]
    recommended_titles = [data['original_title'].iloc[idx] for idx in recommended_indices]
    
    # Print recommendations
    print("\n# Step 7: Print similar movies")
    for i, title in enumerate(recommended_titles, start=1):
        print(f"{i}. {title}")
    
    return recommended_titles

# Step 6: Test the Recommendation System
movie_name = "Mad Max: Fury Road"  # Example movie
recommendations = recommend_movies(movie_name)
from django.shortcuts import render
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

# Fetch movie poster from TMDB API
def fetch_movie_poster(movie_title):
    """
    Fetches the movie poster URL from the TMDB API.
    
    Args:
        movie_title (str): The title of the movie for which the poster is to be fetched.
    
    Returns:
        str: The URL of the movie poster, or None if no poster is found.
    """
    api_key = "10dcb50264517dff89e7037e1241d991"  # Replace with your TMDB API key
    base_url = "https://api.themoviedb.org/3/search/movie"  # TMDB search endpoint
    poster_base_url = "https://image.tmdb.org/t/p/w500"  # Base URL for fetching posters

    # Make a request to the TMDB API
    response = requests.get(base_url, params={"api_key": api_key, "query": movie_title})
    if response.status_code == 200:  # Check for successful response
        data = response.json()
        if data['results']:
            # Get the poster path of the first result
            poster_path = data['results'][0].get('poster_path', '')
            if poster_path:
                return f"{poster_base_url}{poster_path}"
    return None  # Return None if no poster is found

# Preprocess the dataset
def preprocess_data():
    """
    Preprocesses the dataset by selecting relevant features, cleaning data,
    and creating combined features for recommendation.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    csv_path = os.path.join('recommendations', 'static', 'recommendations', 'csv', 'tmdb-movies.csv')
    data = pd.read_csv(csv_path)

    # Features relevant for movie recommendations
    selected_features = ['original_title', 'genres', 'keywords', 'tagline', 'cast', 'director', 'overview',
                         'runtime', 'budget', 'revenue', 'vote_average', 'release_date']
    data = data[selected_features]

    # Fill missing values with empty strings
    for feature in selected_features:
        data[feature] = data[feature].fillna('')

    # Combine features into a single string
    data['combined_features'] = (
        data['genres'] + ' ' +
        data['keywords'] + ' ' +
        data['tagline'] + ' ' +
        data['cast'] + ' ' +
        data['director'] + ' ' +
        data['overview']
    )

    return data

# Preprocess data and calculate similarity once at startup
data = preprocess_data()
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(data['combined_features'])
similarity = cosine_similarity(feature_vectors)

def dataset_statistics():
    """
    Generates basic statistics about the dataset.

    Returns:
        dict: Statistics including total movies, average runtime, budget, revenue, and common genres.
    """
    total_movies = len(data)
    avg_runtime = data['runtime'].astype(float).mean()
    avg_budget = data['budget'].astype(float).mean()
    avg_revenue = data['revenue'].astype(float).mean()

    most_common_genres = data['genres'].str.split('|').explode().value_counts().head(5)

    return {
        'total_movies': total_movies,
        'avg_runtime': round(avg_runtime, 2),
        'avg_budget': round(avg_budget, 2),
        'avg_revenue': round(avg_revenue, 2),
        'most_common_genres': most_common_genres.to_dict()
    }

def recommend_movie(request):
    """
    Handles movie recommendation requests and returns recommended movies.
    """
    stats = dataset_statistics()

    if request.method == "POST":
        movie_name = request.POST.get('movie_name', '').strip()

        # Create a list of all movie titles
        list_of_all_titles = data['original_title'].tolist()

        # Find the closest match
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        if len(find_close_match) == 0:
            return render(request, 'recommendations/index.html', {'error': "No matches found for the given movie.", 'stats': stats})

        # Get the closest match and its index
        close_match = find_close_match[0]
        index_of_movie = data[data['original_title'] == close_match].index[0]

        # Calculate similarity scores
        similarity_score = list(enumerate(similarity[index_of_movie]))

        # Sort movies by similarity
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Fetch top 10 recommendations with metadata
        recommended_movies = []
        for movie in sorted_similar_movies[1:11]:
            index = movie[0]
            title = data['original_title'].iloc[index]
            poster_url = fetch_movie_poster(title)
            overview = data['overview'].iloc[index]
            genres = data['genres'].iloc[index]
            runtime = data['runtime'].iloc[index]
            budget = data['budget'].iloc[index]
            revenue = data['revenue'].iloc[index]
            vote_average = data['vote_average'].iloc[index]
            release_date = data['release_date'].iloc[index]
            keywords = data['keywords'].iloc[index]
            director = data['director'].iloc[index]

            # Append metadata and similarity score
            recommended_movies.append({
                'title': title,
                'poster': poster_url,
                'overview': overview,
                'genres': genres,
                'runtime': runtime,
                'budget': budget,
                'revenue': revenue,
                'rating': vote_average,
                'release_date': release_date,
                'keywords': keywords,
                'director': director,
                'similarity_score': round(movie[1] * 100, 2)  # Convert to percentage
            })

        # Render the results
        return render(request, 'recommendations/index.html', {
            'movie_name': close_match,
            'recommended_movies': recommended_movies,
            'stats': stats
        })

    return render(request, 'recommendations/index.html', {'stats': stats})
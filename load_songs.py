import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np



# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory with JSON song files
directory_path = 'C:\\Users\\andre\\Desktop\\ChurchJSONFiles'

def load_song_data(file_path):
    """Load song data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def generate_embeddings(song_data):
    """Generate an embedding for the given song data."""
    # Concatenate metadata and lyrics into a single text block
    text_to_embed = f"Title: {song_data['metadata']['title']} Artist: {song_data['metadata']['artist']} Language: {song_data['metadata']['language']} Singers: {', '.join(song_data['metadata']['singers'])} Labels: {', '.join(song_data['metadata']['labels'])} Lyrics: {song_data['lyrics']}"
    # Generate and return the embedding
    return model.encode(text_to_embed)



# Function to calculate cosine similarity
def find_top_similar_songs(query_embedding, song_embeddings, top_n=3):
    similarities = []
    for song, embedding in song_embeddings.items():
        similarity = util.pytorch_cos_sim(query_embedding, embedding)
        similarities.append((song, similarity.item()))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top N similar songs
    return similarities[:top_n]


# Loop through each file in the directory
embeddings = {}
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        
        # Load the song data from the JSON file
        song_data = load_song_data(file_path)
        
        # Generate embeddings for the song
        embedding = generate_embeddings(song_data)
        
        # Store the embedding in a dictionary, using filename as the key
        embeddings[filename] = embedding

        # Print progress
        print(f"Generated embedding for {filename}")


user_query = input("Describe a song: ")
query_embedding = model.encode(user_query)

# Find top 3 similar songs
top_songs = find_top_similar_songs(query_embedding, embeddings)

# Print top 3 songs
print("Top 3 songs based on your description:")
for song, similarity in top_songs:
    print(f"{song} with similarity: {similarity}")
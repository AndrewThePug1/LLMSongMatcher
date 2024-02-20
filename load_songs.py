import json
import os
import chromadb

# Specify the path to your persistent storage folder
PERSISTENCE_PATH = os.path.join(os.path.dirname(__file__), "chroma_data") 

# Initialize ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(path=PERSISTENCE_PATH)

# Attempts to retrieve an existing collection or create a new one
try:
    collection = chroma_client.get_collection(name="song_collection")
except Exception as e:
    print("Collection not found, creating a new one...")
    collection = chroma_client.create_collection(name="song_collection")

# Directory with JSON song files
directory_path = os.path.join(os.path.dirname(__file__), "ChurchSongs")

def load_song_data(file_path):
    """Load song data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Loops through each file in the directory and add them to ChromaDB
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        song_data = load_song_data(file_path)
        
        # Adjust metadata to ensure all values are in acceptable formats
        metadata = song_data['metadata']
        # Convert list fields to strings
        metadata['singers'] = ', '.join(metadata.get('singers', []))  # Converts list to str, if exists
        metadata['labels'] = ', '.join(metadata.get('labels', []))  # Adjusts for the 'labels' or similar field
        
        # Add song data to the ChromaDB collection using default embedding
        collection.add(
            documents=[song_data['lyrics']],  # Uses lyrics as the document content for embedding
            metadatas=[metadata],  # Uses the adjusted metadata
            ids=[filename]  # Uses the filename as a unique identifier
        )
      

print("Completed adding songs to ChromaDB collection.")

# Handling user query
user_query = input("Describe a song: ")

# Query method uses text for searching, not direct embeddings
results = collection.query(
    query_texts=[user_query],  # Use the user's query text for searching
    n_results=3  # Number of results to return
)

print("Top 3 songs based on your description:")
if 'ids' in results and 'metadatas' in results and len(results['ids']) > 0:
    for i in range(len(results['ids'][0])):  # Assuming single query, hence [0]
        song_id = results['ids'][0][i]
        song_metadata = results['metadatas'][0][i]
        song_distance = results['distances'][0][i]
        print(f"Song ID: {song_id}, Title: {song_metadata['title']}, Artist: {song_metadata['artist']}, Distance: {song_distance}")
else:
    print("No results found.")

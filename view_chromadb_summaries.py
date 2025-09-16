from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

from ingest.db_handler import ChromaDBHandler

# Connect to ChromaDB
chroma = ChromaDBHandler()

# Get the float_embeddings collection
try:
    collection = chroma.client.get_collection("float_embeddings")
    
    # Get all documents to see the semantic summaries
    results = collection.get(include=["documents", "metadatas"])
    
    print(f"Found {len(results['documents'])} semantic summaries in ChromaDB:")
    print("=" * 80)
    
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
        print(f"\n{i+1}. Float ID: {metadata.get('float_id', 'Unknown')}")
        print(f"   Date: {metadata.get('date', 'Unknown')}")
        print(f"   Location: {metadata.get('lat', 'N/A')}°, {metadata.get('lon', 'N/A')}°")
        print(f"   Profiles: {metadata.get('profiles', 'N/A')}")
        print(f"   Summary: {doc}")
        print("-" * 80)

except Exception as e:
    print(f"Error: {e}")

finally:
    chroma.close()
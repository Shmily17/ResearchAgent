import chromadb
from chromadb.config import Settings as ChromaSettings

CHROMA_DB_DIR = "./chroma_db"
student_id_for_collection = "xiaoi_ming"

collection_name = f"research_papers_{student_id_for_collection}"

client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=ChromaSettings(anonymized_telemetry=False))
try:
    print(f"Attempting to delete collection: {collection_name}")
    client.delete_collection(name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
except Exception as e:
    # It might raise an exception if the collection doesn't exist, which is fine.

    print(f"Error deleting collection '{collection_name}' (or it didn't exist): {e}")

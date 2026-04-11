import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = "vector_db_store"
COLLECTION_NAME = "india_policy_reports"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory=VECTOR_DB_DIR
    )
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_policy_context(query, k=2):

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    docs = results["documents"][0]

    return "\n\n".join(docs)
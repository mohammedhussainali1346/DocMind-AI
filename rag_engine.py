from sentence_transformers import SentenceTransformer
import chromadb
import hashlib
from db_client import chroma_client

# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 2. Connect to the persistent ChromaDB database
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(path="./chroma_db")

document_collection = chroma_client.get_or_create_collection(
    name="document_intelligence"
)


# --------------------------------------------------
# 3. Split document text into overlapping chunks
# --------------------------------------------------


def create_chunks(document, chunk_size=500, overlap=50):

    # Validate chunk settings
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or not isinstance(overlap, int)
        or isinstance(overlap, bool)
        or chunk_size <= 0
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_size must be positive and overlap "
            "must be non-negative and smaller than chunk_size."
        )

    # Get the cleaned text
    text = document["cleaned_text"]

    # Split text into words
    words = text.split()

    # Return no chunks for an empty document
    if not words:
        return []

    chunks = []

    # Calculate the distance between chunk starts
    step = chunk_size - overlap

    # Create overlapping chunks
    for start in range(0, len(words), step):

        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            continue

        chunks.append(
            {
                "chunk_text": " ".join(chunk_words),
                "file_name": document["file_name"],
                "file_type": document["file_type"],
                "start_word": start,
                "end_word": start + len(chunk_words),
                "content_hash": document["content_hash"],
            }
        )

        # Stop after processing the final chunk
        if start + chunk_size >= len(words):
            break

    return chunks


# --------------------------------------------------
# 4. Store chunks and replace outdated chunks safely
# --------------------------------------------------


def store_chunks(chunks, file_name=None):

    # Support an empty document when its filename is known
    if not chunks:

        if file_name is not None:
            document_collection.delete(where={"file_name": file_name})

        return 0

    # Identify every document represented in this batch
    file_names = {chunk["file_name"] for chunk in chunks}

    # Prepare all chunk texts
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]

    # Generate embeddings before changing the database
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()

    ids = []
    metadatas = []

    # Track new IDs belonging to each document
    new_ids_by_file = {name: set() for name in file_names}

    # Track chunk numbers separately for each filename
    chunk_counts = {name: 0 for name in file_names}

    # Prepare IDs and metadata
    for chunk in chunks:

        file_name_for_chunk = chunk["file_name"]

        unique_text = (
            file_name_for_chunk
            + chunk["content_hash"]
            + str(chunk["start_word"])
            + str(chunk["end_word"])
        )

        chunk_id = hashlib.sha256(unique_text.encode("utf-8")).hexdigest()

        ids.append(chunk_id)

        new_ids_by_file[file_name_for_chunk].add(chunk_id)

        chunk_number = chunk_counts[file_name_for_chunk]
        chunk_counts[file_name_for_chunk] += 1

        metadatas.append(
            {
                "file_name": file_name_for_chunk,
                "file_type": chunk["file_type"],
                "chunk_number": chunk_number,
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "content_hash": chunk["content_hash"],
            }
        )

    # Record existing IDs before inserting the new version
    old_ids_by_file = {}

    for name in file_names:

        existing = document_collection.get(where={"file_name": name}, include=[])

        old_ids_by_file[name] = set(existing["ids"])

    # Insert or update the new chunks first.
    # If embedding generation fails, the old chunks remain.
    document_collection.upsert(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Remove outdated IDs only after the upsert succeeds
    for name in file_names:

        outdated_ids = old_ids_by_file[name] - new_ids_by_file[name]

        if outdated_ids:
            document_collection.delete(ids=list(outdated_ids))

    return len(chunks)


# --------------------------------------------------
# 5. Retrieve relevant chunks for a question
# --------------------------------------------------


def retrieve_chunks(question, n_results=3, distance_threshold=1.5):

    # Validate the requested number of results
    if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results <= 0:
        raise ValueError("n_results must be a positive integer.")

    # Validate the distance threshold
    if distance_threshold is not None:
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative or None.")

    # Check whether the database contains chunks
    total_chunks = document_collection.count()

    if total_chunks == 0:
        return []

    # Validate the question
    if not isinstance(question, str) or not question.strip():
        return []

    # Convert the question into an embedding
    question_embedding = embedding_model.encode(question.strip()).tolist()

    # Retrieve the closest matching chunks
    results = document_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    # Process the retrieved results
    for i, chunk_text in enumerate(results["documents"][0]):

        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        # Apply the optional distance threshold
        if distance_threshold is not None and distance > distance_threshold:
            continue

        retrieved_chunks.append(
            {
                "chunk_text": chunk_text,
                "file_name": metadata["file_name"],
                "file_type": metadata["file_type"],
                "chunk_number": metadata["chunk_number"],
                "distance": distance,
            }
        )

    return retrieved_chunks

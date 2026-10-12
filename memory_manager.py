import uuid

from db_client import chroma_client
from rag_engine import embedding_model

# --------------------------------------------------
# 1. Connect to the user memory collection
# --------------------------------------------------

# Uses the same configured database as rag_engine.py.
memory_collection = chroma_client.get_or_create_collection(name="user_memory")


# --------------------------------------------------
# 2. Save a new memory
# --------------------------------------------------


def save_memory(memory_text):

    # Validate the memory text.
    if not isinstance(memory_text, str):
        return False

    memory_text = memory_text.strip()

    if not memory_text or memory_text.lower() == "no memory":
        return False

    # Convert the memory into an embedding.
    memory_embedding = embedding_model.encode(memory_text).tolist()

    # Check whether a similar memory already exists.
    existing_count = memory_collection.count()

    if existing_count > 0:

        results = memory_collection.query(
            query_embeddings=[memory_embedding],
            n_results=1,
            include=["distances"],
        )

        distances = results.get("distances", [[]])[0]

        # Skip a very similar memory.
        if distances and distances[0] < 0.25:
            return False

    # Generate a unique memory ID.
    memory_id = str(uuid.uuid4())

    # Save the memory and its embedding.
    memory_collection.add(
        ids=[memory_id],
        documents=[memory_text],
        embeddings=[memory_embedding],
    )

    return True


# --------------------------------------------------
# 3. Retrieve memories relevant to a user message
# --------------------------------------------------


def retrieve_memory(user_message, n_results=3):

    # Validate the requested result count.
    if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results <= 0:
        raise ValueError("n_results must be a positive integer.")

    # Validate the user's message.
    if not isinstance(user_message, str) or not user_message.strip():
        return []

    total_memories = memory_collection.count()

    if total_memories == 0:
        return []

    # Embed the user's message.
    query_embedding = embedding_model.encode(user_message.strip()).tolist()

    # Search for similar memories.
    results = memory_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total_memories),
        include=[
            "documents",
            "distances",
        ],
    )

    memories = []

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for memory, distance in zip(documents, distances):

        if distance < 1.5:
            memories.append(
                {
                    "memory": memory,
                    "distance": distance,
                }
            )

    return memories

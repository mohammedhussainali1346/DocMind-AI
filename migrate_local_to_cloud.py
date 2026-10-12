import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

# Connect to your existing local database.
local_client = chromadb.PersistentClient(path="./chroma_db")

# Connect to your hosted Chroma Cloud database.
cloud_client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)


def migrate_collection(collection_name):

    # Read the source collection.
    source = local_client.get_collection(name=collection_name)

    # Create or connect to the destination collection.
    destination = cloud_client.get_or_create_collection(name=collection_name)

    # Copy records in batches.
    batch_size = 100
    offset = 0
    migrated = 0

    while True:

        batch = source.get(
            limit=batch_size,
            offset=offset,
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

        ids = batch["ids"]

        if not ids:
            break

        records = {
            "ids": ids,
            "documents": batch["documents"],
            "embeddings": batch["embeddings"].tolist(),
        }

        # Memory records may not have metadata.
        metadatas = batch["metadatas"]

        if metadatas and all(metadata is not None for metadata in metadatas):
            records["metadatas"] = metadatas

        # Preserve IDs, documents, and existing embeddings.
        destination.upsert(**records)

        migrated += len(ids)
        offset += len(ids)

    print(f"{collection_name}: migrated {migrated} records.")


if __name__ == "__main__":

    for name in [
        "document_intelligence",
        "user_memory",
    ]:

        migrate_collection(name)

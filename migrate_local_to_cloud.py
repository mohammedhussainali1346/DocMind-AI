import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

# Connect to the existing local database.
local_client = chromadb.PersistentClient(path="./chroma_db")

# Connect to Chroma Cloud.
cloud_client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)


def migrate_collection(collection_name):
    print(f"\nProcessing collection: {collection_name}")

    # Get the existing local collection.
    source = local_client.get_collection(name=collection_name)

    # Get or create the corresponding cloud collection.
    destination = cloud_client.get_or_create_collection(name=collection_name)

    # Count records before migration.
    local_count = source.count()
    cloud_count_before = destination.count()

    print(f"Local records: {local_count}")
    print(f"Cloud records before migration: {cloud_count_before}")

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
            "embeddings": batch["embeddings"].tolist(),
        }

        # Add documents when available.
        if batch["documents"] is not None:
            records["documents"] = batch["documents"]

        # Add metadata when available.
        if batch["metadatas"] is not None:
            records["metadatas"] = batch["metadatas"]

        # Copy records without deleting local data.
        destination.upsert(**records)

        migrated += len(ids)
        offset += len(ids)

    # Verify the cloud collection after migration.
    cloud_count_after = destination.count()

    print(f"Records processed: {migrated}")
    print(f"Cloud records after migration: {cloud_count_after}")

    if cloud_count_after >= local_count:
        print(f"SUCCESS: {collection_name} migration verified.")
    else:
        print(f"WARNING: Check the migration of {collection_name}.")


if __name__ == "__main__":
    for name in ["document_intelligence", "user_memory"]:
        migrate_collection(name)

    print("\nMigration process finished.")

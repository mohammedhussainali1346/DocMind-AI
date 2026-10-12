import os

import chromadb
from dotenv import load_dotenv

# Load local environment variables when developing.
load_dotenv()


# Read the hosted database credentials.
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


# Use Chroma Cloud when its API key is configured.
if CHROMA_API_KEY:

    if not CHROMA_TENANT or not CHROMA_DATABASE:
        raise RuntimeError(
            "Set CHROMA_TENANT and CHROMA_DATABASE " "along with CHROMA_API_KEY."
        )

    chroma_client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

# Otherwise, use the existing local database.
else:

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

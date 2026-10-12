import os

import chromadb
import streamlit as st
from dotenv import load_dotenv

# Load .env variables during local development.
load_dotenv()


def get_setting(name, default=None):
    """Read a setting from environment variables or Streamlit secrets."""

    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name, default)
        return value
    except (FileNotFoundError, RuntimeError):
        # Streamlit secrets may not exist during standalone scripts.
        return default


CHROMA_API_KEY = get_setting("CHROMA_API_KEY")
CHROMA_TENANT = get_setting(
    "CHROMA_TENANT",
    "c891f530-f6f0-442c-add3-1f9386d028cc",
)
CHROMA_DATABASE = get_setting(
    "CHROMA_DATABASE",
    "docmind-ai",
)

if CHROMA_API_KEY:
    # Use Chroma Cloud when credentials are configured.
    chroma_client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )
else:
    # Fall back to the local database.
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

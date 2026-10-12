from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pandas as pd
from pypdf import PdfReader
from docx import Document
from pptx import Presentation

# --------------------------------------------------
# 1. Extract text from PDF files
# --------------------------------------------------


def extract_pdf(file_path):

    reader = PdfReader(file_path)
    text = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        text.append(f"Page {page_number}:\n{page_text}")

    return "\n".join(text)


# --------------------------------------------------
# 2. Extract text from Word documents
# --------------------------------------------------


def extract_docx(file_path):

    document = Document(file_path)
    text = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            text.append(paragraph.text)

    for table in document.tables:

        for row in table.rows:

            text.append(" | ".join(cell.text for cell in row.cells))

    return "\n".join(text)


# --------------------------------------------------
# 3. Extract text from TXT and Markdown files
# --------------------------------------------------


def extract_text(file_path):

    return Path(file_path).read_text(
        encoding="utf-8",
        errors="replace",
    )


# --------------------------------------------------
# 4. Extract data from CSV files
# --------------------------------------------------


def extract_csv(file_path):

    dataframe = pd.read_csv(file_path)

    return dataframe.to_string(index=False)


# --------------------------------------------------
# 5. Extract data from Excel files
# --------------------------------------------------


def extract_excel(file_path):

    sheets = pd.read_excel(
        file_path,
        sheet_name=None,
    )

    text = []

    for sheet_name, dataframe in sheets.items():

        text.append(f"Sheet: {sheet_name}")

        text.append(dataframe.to_string(index=False))

    return "\n".join(text)


# --------------------------------------------------
# 6. Extract text from PowerPoint files
# --------------------------------------------------


def extract_pptx(file_path):

    presentation = Presentation(file_path)
    text = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):

        text.append(f"Slide {slide_number}:")

        for shape in slide.shapes:

            if shape.has_text_frame:
                text.append(shape.text)

            if shape.has_table:

                for row in shape.table.rows:

                    text.append(" | ".join(cell.text for cell in row.cells))

    return "\n".join(text)


# --------------------------------------------------
# 7. Extract data from JSON files
# --------------------------------------------------


def extract_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------
# 8. Select the correct extractor
# --------------------------------------------------


def extract_document(file_path):

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".txt": extract_text,
        ".md": extract_text,
        ".csv": extract_csv,
        ".xlsx": extract_excel,
        ".pptx": extract_pptx,
        ".json": extract_json,
    }

    if extension not in extractors:

        raise ValueError(f"Unsupported file type: {extension}")

    return extractors[extension](file_path)


# --------------------------------------------------
# 9. Clean extracted text
# --------------------------------------------------


def clean_text(text):

    # Ensure the input is text.
    if not isinstance(text, str):
        text = str(text)

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFC", text)

    # Remove unwanted control characters.
    # Preserve newlines and tabs.
    text = "".join(
        char for char in text if (unicodedata.category(char) != "Cc" or char in "\n\t")
    )

    # Replace tabs and repeated horizontal whitespace
    # with a single space, preserving line breaks.
    text = re.sub(
        r"[^\S\n]+",
        " ",
        text,
    )

    # Remove spaces at the beginning of each line.
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    # Reduce three or more consecutive newlines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# --------------------------------------------------
# 10. Clean documents and identify duplicate content
# --------------------------------------------------


def preprocess_documents(documents):

    processed_documents = []

    # Track duplicate content within this batch.
    seen_hashes = set()

    for document in documents:

        # Get the original extracted text.
        original_text = document["text"]

        # Clean the extracted text.
        cleaned = clean_text(original_text)

        # Generate a hash of the cleaned content.
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        # Detect duplicates within this batch.
        is_duplicate = content_hash in seen_hashes

        seen_hashes.add(content_hash)

        # Preserve the original metadata and add new fields.
        processed_documents.append(
            {
                **document,
                "cleaned_text": cleaned,
                "character_count": len(cleaned),
                "is_empty": not bool(cleaned.strip()),
                "is_duplicate": is_duplicate,
                "content_hash": content_hash,
            }
        )

    return processed_documents

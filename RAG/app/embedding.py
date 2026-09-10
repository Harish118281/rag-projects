import os
import pickle
import hashlib

import faiss

from sentence_transformers import SentenceTransformer


VECTOR_FOLDER = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_FOLDER,
    "faiss.index"
)

CHUNKS_FILE = os.path.join(
    VECTOR_FOLDER,
    "chunks.pkl"
)

PDF_HASH_FILE = os.path.join(
    VECTOR_FOLDER,
    "source_pdf.sha256"
)


def get_file_hash(file_path):

    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:

        for block in iter(lambda: file.read(8192), b""):

            hasher.update(block)

    return hasher.hexdigest()


def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def create_embeddings(
    model,
    chunks
):

    return model.encode(
        chunks,
        convert_to_numpy=True
    )


def save_vector_database(
    embeddings,
    chunks,
    pdf_path
):

    os.makedirs(
        VECTOR_FOLDER,
        exist_ok=True
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    with open(PDF_HASH_FILE, "w", encoding="utf-8") as file:

        file.write(
            get_file_hash(pdf_path)
        )


def load_vector_database():

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        CHUNKS_FILE,
        "rb"
    ) as file:

        chunks = pickle.load(
            file
        )

    return index, chunks


def vector_database_exists(pdf_path):

    required_files_exist = (
        os.path.exists(INDEX_FILE)
        and os.path.exists(CHUNKS_FILE)
        and os.path.exists(PDF_HASH_FILE)
    )

    if not required_files_exist:

        return False

    with open(PDF_HASH_FILE, "r", encoding="utf-8") as file:

        saved_pdf_hash = file.read().strip()

    return saved_pdf_hash == get_file_hash(pdf_path)



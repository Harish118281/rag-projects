from app.loader import load_pdf

from app.chunk import create_chunks

from app.embedding import (
    load_embedding_model,
    create_embeddings,
    save_vector_database,
    load_vector_database,
    vector_database_exists
)

from app.retrievel import (
    retrieve_context
)

from app.generator import (
    generate_answer
)


embedding_model = load_embedding_model()

PDF_PATH = "documents/book.pdf"

if not vector_database_exists(PDF_PATH):

    print("Creating vector database...")

    full_text = load_pdf(
        PDF_PATH
    )

    chunks = create_chunks(
        full_text
    )

    embeddings = create_embeddings(
        embedding_model,
        chunks
    )

    save_vector_database(
        embeddings,
        chunks,
        PDF_PATH
    )

    print("Vector database created.")

else:

    print("Using existing vector database.")


index, chunks = load_vector_database()


print("\nRAG chatbot is ready.")


while True:

    question = input(
        "\nQuestion: "
    )

    if question.lower() == "exit":

        break

    context = retrieve_context(
        question,
        embedding_model,
        index,
        chunks
    )

    if context is None:

        answer = generate_answer(
                question,
                context
            )

    else :
        answer = generate_answer(
            question,
            context
        )

    print("\nAnswer:\n")

    print(answer)

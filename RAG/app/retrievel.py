def retrieve_context(
    question,
    embedding_model,
    index,
    chunks,
    distance_threshold=1.0
):

    question_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_vector,
        len(chunks)
    )

    matching_chunks = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if distance <= distance_threshold:

            matching_chunks.append(
                chunks[idx]
            )

    if not matching_chunks:

        return None

    return "\n\n".join(
        matching_chunks
    )
import os
from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_answer(question, context):

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not present in the context,switch to general knowledge and answer to questions

CONTEXT:

{context}

QUESTION:

{question}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

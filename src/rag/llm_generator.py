import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_answer(question, context):
    """
    Generate an evidence-grounded answer
    using the retrieved medical context.
    """

    prompt = f"""
You are a breast cancer clinical support and patient education assistant.

Answer the user's question using ONLY the evidence provided below.

IMPORTANT RULES:

1. Do not invent medical facts.
2. Do not make a diagnosis.
3. Do not prescribe or change treatment.
4. If the evidence is insufficient, clearly say so.
5. For personal medical decisions, recommend consultation
   with a qualified healthcare professional.
6. Keep the answer clear and evidence-grounded.
7. Mention the relevant sources when possible.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Provide a clear answer based on the retrieved evidence.
"""

    interaction = client.interactions.create(
        model="gemini-3.7-flash",
        input=prompt
    )

    return interaction.output_text


if __name__ == "__main__":

    print("=" * 60)
    print("GEMINI LLM TEST")
    print("=" * 60)

    question = "What is breast cancer?"

    context = """
Breast cancer is a disease in which cells in the breast
grow uncontrollably. Breast cancer can begin in different
parts of the breast.
"""

    answer = generate_answer(
        question,
        context
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)
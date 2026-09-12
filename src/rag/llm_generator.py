import os

from dotenv import load_dotenv
from google import genai
from user_mode import UserMode, get_mode_instructions

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_answer(question, context, mode=UserMode.PATIENT):
    """
    Generate an evidence-grounded answer using Gemini.

    Parameters:
        question: User's question
        context: Retrieved evidence
        mode: UserMode.PATIENT or UserMode.DOCTOR
    """

    # Get instructions according to the selected user mode
    mode_instructions = get_mode_instructions(mode)

    prompt = f"""
You are a breast cancer clinical support and patient education assistant.

{mode_instructions}

Your task is to answer the user's question using ONLY the evidence
provided below.

IMPORTANT EVIDENCE RULES:

1. Use ONLY information supported by the provided evidence.
2. Do not invent, assume, or guess medical information.
3. Do not make a diagnosis.
4. Do not prescribe treatment.
5. Do not recommend changing a patient's treatment.
6. If the evidence is insufficient, say:
   "The available evidence is insufficient to answer this reliably."
7. Do not use outside medical knowledge.
8. Preserve important medical terminology accurately.
9. For personal medical decisions, recommend consultation with
   a qualified healthcare professional.

CITATION RULES:

1. Every important medical claim must cite the evidence used.
2. Use the exact evidence identifier provided in the context.
3. Cite sources using this format:

   [Evidence 1]

   or

   [Evidence 1, Evidence 3]

4. NEVER write citations such as:
   "Source 1"
   "Source 2"
   "Source 3"

5. NEVER invent an evidence number.
6. Only cite evidence that actually supports the statement.
7. In the Sources section, list the actual source title and page
   associated with the evidence.

ANSWER FORMAT:

### Answer

Give a clear and concise answer to the user's question.

Use bullet points when appropriate.

### Evidence

Briefly explain the important evidence used to answer the question.
Include the appropriate [Evidence X] citations.

### Sources

List only the evidence sources actually used.

Example:

- Treatment for Breast Cancer - NCI.pdf, Page 1 [Evidence 1]
- Breast cancer fact sheet.pdf, Page 4 [Evidence 3]

### Medical Safety

If the question involves diagnosis, treatment decisions,
medications, symptoms, or a personal medical situation,
include a brief recommendation to consult a qualified
healthcare professional.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}
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
[EVIDENCE 1]

Source: Breast Cancer - NCI.pdf
Page: 1

Content:
Breast cancer is a disease in which cells in the breast
grow uncontrollably.
"""

    print("\nTesting Patient Mode...")
    patient_answer = generate_answer(
        question,
        context,
        mode=UserMode.PATIENT
    )

    print("\nPATIENT MODE ANSWER:")
    print(patient_answer)

    print("\n" + "=" * 60)

    print("\nTesting Doctor Mode...")
    doctor_answer = generate_answer(
        question,
        context,
        mode=UserMode.DOCTOR
    )

    print("\nDOCTOR MODE ANSWER:")
    print(doctor_answer)
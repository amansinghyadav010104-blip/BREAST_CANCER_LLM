# -----------------------------------------
# Query / Intent Router
# -----------------------------------------

def classify_query(query):
    """
    Classify the user's question into a basic intent.
    """

    query = query.lower().strip()

    # Patient education
    patient_keywords = [
        "what is",
        "meaning",
        "explain",
        "symptoms",
        "signs",
        "side effects",
        "what does",
        "tell me about"
    ]

    # Clinical / medical information
    clinical_keywords = [
        "treatment",
        "therapy",
        "chemotherapy",
        "radiation",
        "surgery",
        "hormone therapy",
        "immunotherapy",
        "targeted therapy",
        "diagnosis",
        "stage",
        "staging",
        "biopsy",
        "receptor",
        "er positive",
        "pr positive",
        "her2",
        "triple negative"
    ]

    # Research / literature
    research_keywords = [
        "research",
        "study",
        "paper",
        "literature",
        "clinical trial",
        "evidence",
        "latest research"
    ]

    # Check research intent first
    if any(
        keyword in query
        for keyword in research_keywords
    ):
        return "research"

    # Check clinical intent
    if any(
        keyword in query
        for keyword in clinical_keywords
    ):
        return "clinical"

    # Check patient education intent
    if any(
        keyword in query
        for keyword in patient_keywords
    ):
        return "patient_education"

    # General breast cancer question
    return "general"


# -----------------------------------------
# Test Router
# -----------------------------------------

if __name__ == "__main__":

    print("Breast Cancer Query Router")
    print("=" * 50)

    while True:

        query = input(
            "\nEnter your question "
            "(type 'exit' to stop): "
        )

        if query.lower() == "exit":
            break

        intent = classify_query(query)

        print(
            "Detected Intent:",
            intent
        )
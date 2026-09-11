def check_safety(query):
    """
    Check whether a breast cancer question requires
    additional medical safety handling.
    """

    query = query.lower().strip()

    diagnosis_keywords = [
        "do i have cancer",
        "do i have breast cancer",
        "am i suffering from",
        "is this cancer",
        "diagnose me",
        "diagnosis for me"
    ]

    medication_keywords = [
        "what dose",
        "dosage",
        "how much medicine",
        "how much medication",
        "prescribe",
        "prescription"
    ]

    treatment_decision_keywords = [
        "which treatment should i choose",
        "what treatment should i take",
        "should i take",
        "should i start",
        "should i stop",
        "should i change",
        "should i switch",
        "best treatment for me",
        "what treatment is best for me"
    ]

    # 1. Diagnosis check
    if any(keyword in query for keyword in diagnosis_keywords):
        return {
            "safe": False,
            "risk_type": "diagnosis",
            "message": (
                "I cannot diagnose breast cancer. "
                "A qualified healthcare professional must evaluate "
                "your symptoms, medical history, examination, and "
                "appropriate diagnostic tests."
            )
        }

    # 2. Medication check
    if any(keyword in query for keyword in medication_keywords):
        return {
            "safe": False,
            "risk_type": "medication",
            "message": (
                "I cannot prescribe medication or provide a "
                "personalized medication dose. Please consult "
                "a qualified healthcare professional."
            )
        }

    # 3. Treatment decision check
    if any(keyword in query for keyword in treatment_decision_keywords):
        return {
            "safe": False,
            "risk_type": "treatment_decision",
            "message": (
                "I cannot choose, change, start, or stop a cancer "
                "treatment for you. Treatment decisions should be "
                "made with your qualified cancer care team."
            )
        }

    return {
        "safe": True,
        "risk_type": None,
        "message": None
    }


if __name__ == "__main__":

    print("=" * 60)
    print("BREAST CANCER SAFETY GUARDRAIL TEST")
    print("=" * 60)

    test_questions = [
        "What is breast cancer?",
        "What are the treatment options for breast cancer?",
        "Do I have breast cancer?",
        "Should I stop chemotherapy?",
        "What dose of medicine should I take?"
    ]

    for question in test_questions:

        result = check_safety(question)

        print("\nQuestion:", question)
        print("Safe:", result["safe"])
        print("Risk type:", result["risk_type"])

        if result["message"]:
            print("Safety message:", result["message"])
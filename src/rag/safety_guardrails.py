import re

from .answer_language import AnswerLanguage
from .cancer_scope import check_cancer_scope


MESSAGES = {
    "diagnosis": (
        "I cannot diagnose breast cancer. Please consult a qualified healthcare professional.",
        "मैं स्तन कैंसर का निदान नहीं कर सकता। कृपया किसी योग्य स्वास्थ्य पेशेवर से सलाह लें।",
    ),
    "medication": (
        "I cannot prescribe medication or provide a personal dose. Please consult a qualified healthcare professional.",
        "मैं दवा लिख या व्यक्तिगत खुराक तय नहीं कर सकता। कृपया किसी योग्य स्वास्थ्य पेशेवर से सलाह लें।",
    ),
    "treatment_decision": (
        "I cannot choose, start, stop, or change personal cancer treatment. Please discuss this with your cancer care team.",
        "मैं आपके लिए कैंसर उपचार चुनने, शुरू करने, रोकने या बदलने का निर्णय नहीं कर सकता। कृपया अपनी कैंसर देखभाल टीम से बात करें।",
    ),
    "unrelated_cancer": (
        "This assistant is limited to breast cancer. I cannot answer questions about other cancers.",
        "यह सहायक केवल स्तन कैंसर के बारे में जानकारी देता है। मैं अन्य कैंसर के प्रश्नों का उत्तर नहीं दे सकता।",
    ),
    "breast_cancer_only": (
        "This assistant is limited to breast cancer questions.",
        "यह सहायक केवल स्तन कैंसर से जुड़े प्रश्नों के लिए है।",
    ),
    "insufficient_evidence": (
        "The available breast cancer evidence is insufficient to answer this reliably.",
        "उपलब्ध स्तन कैंसर साक्ष्य इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त नहीं हैं।",
    ),
    "generation_unavailable": (
        "Relevant breast cancer evidence was retrieved, but a reliable answer could not be generated right now. Please try again later. For treatment decisions, consult your oncologist.",
        "स्तन कैंसर से संबंधित साक्ष्य मिले, लेकिन अभी विश्वसनीय उत्तर तैयार नहीं हो सका। कृपया बाद में फिर कोशिश करें। उपचार संबंधी निर्णयों के लिए अपने कैंसर विशेषज्ञ से सलाह लें।",
    ),
    "emergency": (
        "If this may be a medical emergency, contact your local emergency services or go to the nearest emergency department now. Do not wait for this assistant.",
        "यदि यह चिकित्सा आपातस्थिति हो सकती है, तो अभी स्थानीय आपातकालीन सेवाओं से संपर्क करें या निकटतम आपातकालीन विभाग जाएँ। इस सहायक के उत्तर की प्रतीक्षा न करें।",
    ),
}


def localized_safety_message(risk_type, answer_language=AnswerLanguage.ENGLISH):
    try:
        language = AnswerLanguage(answer_language)
    except ValueError:
        language = AnswerLanguage.ENGLISH
    english, hindi = MESSAGES[risk_type]
    if language == AnswerLanguage.HINDI:
        return hindi
    if language == AnswerLanguage.BILINGUAL:
        return f"{english}\n\n{hindi}"
    return english


def check_safety(query, answer_language=AnswerLanguage.ENGLISH):
    """
    Check whether a breast cancer question requires
    additional medical safety handling.
    """

    query = query.casefold().strip()

    emergency_keywords = (
        "medical emergency", "emergency", "urgent medical help", "can't breathe",
        "cannot breathe", "difficulty breathing", "severe bleeding", "unconscious",
        "चिकित्सा आपातस्थिति", "आपातकाल", "सांस लेने में कठिनाई", "साँस लेने में कठिनाई",
        "सांस नहीं ले", "साँस नहीं ले", "बेहोश", "बहुत खून बह",
    )
    if any(keyword in query for keyword in emergency_keywords):
        risk_type = "emergency"
        return {
            "safe": False,
            "risk_type": risk_type,
            "message": localized_safety_message(risk_type, answer_language),
        }

    diagnosis_keywords = [
        "do i have cancer",
        "do i have breast cancer",
        "am i suffering from",
        "is this cancer",
        "diagnose me",
        "diagnosis for me",
        "क्या मुझे स्तन कैंसर है", "क्या मुझे कैंसर है", "मुझे कैंसर है क्या",
        "क्या यह कैंसर है", "क्या यह स्तन कैंसर है", "मुझे स्तन कैंसर है क्या",
        "kya mujhe breast cancer hai", "kya mujhe stan cancer hai",
    ]
    personal_diagnosis_patterns = (
        (
            r"\b(?:could|might|may|is|are)\s+(?:this|that|my)\b.{0,80}"
            r"\b(?:lump|mass|symptom|biopsy|mammogram|test|scan|report|result)\b"
            r".{0,80}\b(?:be|cancer|malignant)\b"
        ),
        (
            r"\b(?:does|do)\s+(?:this|that|my)\b.{0,80}"
            r"\b(?:lump|mass|symptom|biopsy|mammogram|test|scan|report|result)\b"
            r".{0,80}\b(?:mean|show|indicate|suggest)\b.{0,40}\b(?:cancer|malignant)\b"
        ),
        (
            r"(?:क्या|क्या यह|क्या मेरी|क्या मेरे).{0,60}"
            r"(?:गांठ|लक्षण|बायोप्सी|रिपोर्ट|जांच|मैमोग्राम).{0,60}"
            r"(?:कैंसर|घातक)"
        ),
    )

    medication_keywords = [
        "what dose",
        "dosage",
        "how much medicine",
        "how much medication",
        "prescribe",
        "prescription",
        "दवा की खुराक", "कितनी दवा", "दवा कितनी लेनी चाहिए", "दवा लिखें",
        "कौन सी दवा लूं", "कितनी दवा लूं",
    ]

    treatment_decision_keywords = [
        "which treatment should i choose",
        "what treatment should i choose",
        "what treatment should i take",
        "should i take",
        "should i start",
        "should i stop",
        "should i change",
        "should i switch",
        "best treatment for me",
        "what treatment is best for me",
        "मुझे कौन सा इलाज लेना चाहिए", "मेरा इलाज बदलना चाहिए", "इलाज बंद करूं",
        "दवा बंद करूं", "इलाज शुरू करूं", "इलाज बदलूं",
    ]

    # 1. Diagnosis check
    if any(keyword in query for keyword in diagnosis_keywords) or any(
        re.search(pattern, query) for pattern in personal_diagnosis_patterns
    ):
        risk_type = "diagnosis"
        return {
            "safe": False,
            "risk_type": risk_type,
            "message": localized_safety_message(risk_type, answer_language),
        }

    # 2. Medication check
    if any(keyword in query for keyword in medication_keywords):
        risk_type = "medication"
        return {
            "safe": False,
            "risk_type": risk_type,
            "message": localized_safety_message(risk_type, answer_language),
        }

    # 3. Treatment decision check
    if any(keyword in query for keyword in treatment_decision_keywords):
        risk_type = "treatment_decision"
        return {
            "safe": False,
            "risk_type": risk_type,
            "message": localized_safety_message(risk_type, answer_language),
        }

    scope = check_cancer_scope(query)
    if not scope["in_scope"]:
        risk_type = scope["reason"]
        return {
            "safe": False,
            "risk_type": risk_type,
            "message": localized_safety_message(risk_type, answer_language),
        }

    return {
        "safe": True,
        "risk_type": None,
        "message": None
    }


def insufficient_evidence_message(answer_language=AnswerLanguage.ENGLISH):
    return localized_safety_message("insufficient_evidence", answer_language)


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

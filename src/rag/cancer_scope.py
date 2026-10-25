import re


BREAST_MARKERS = (
    "breast cancer", "breast carcinoma", "mammary carcinoma", "mammary cancer",
    "mastectom", "lumpectom", "mammogram", "mammograph", "her2", "er positive",
    "er-positive", "pr positive", "pr-positive", "triple negative", "triple-negative",
    "sentinel node", "tamoxifen", "trastuzumab", "स्तन कैंसर", "स्तन का कैंसर",
    "ब्रेस्ट कैंसर", "स्तन में गांठ", "स्तन की गांठ", "स्तन में दर्द", "मैमोग्राम",
    "मैमोग्राफी", "मास्टेक्टॉमी", "लम्पेक्टॉमी",
)

OTHER_CANCER_MARKERS = (
    "lung cancer", "prostate cancer", "colon cancer", "colorectal cancer",
    "ovarian cancer", "pancreatic cancer", "brain cancer", "skin cancer",
    "cervical cancer", "stomach cancer", "liver cancer", "leukemia", "lymphoma",
    "फेफड़ों का कैंसर", "प्रोस्टेट कैंसर", "आंत का कैंसर", "अंडाशय का कैंसर",
    "अग्न्याशय का कैंसर", "मस्तिष्क का कैंसर", "त्वचा का कैंसर", "गर्भाशय ग्रीवा का कैंसर",
)

CANCER_MARKERS = (
    "cancer", "carcinoma", "oncology", "tumour", "tumor", "neoplasm",
    "कैंसर", "कार्सिनोमा", "ऑन्कोलॉजी", "ट्यूमर",
)


def normalize_breast_spelling(text):
    return re.sub(r"\bbrest\b", "breast", text, flags=re.IGNORECASE)


def is_breast_cancer_related(text, source=""):
    combined = f"{source} {text}".casefold()
    if any(marker in combined for marker in OTHER_CANCER_MARKERS):
        return False
    if any(marker in combined for marker in BREAST_MARKERS):
        return True
    if "breast" in combined and any(
        marker in combined for marker in ("lump", "pain", "screening", "treatment", "stage")
    ):
        return True
    return False


def check_cancer_scope(query):
    normalized = normalize_breast_spelling(query).casefold()
    if any(marker in normalized for marker in OTHER_CANCER_MARKERS):
        return {"in_scope": False, "reason": "unrelated_cancer"}
    if is_breast_cancer_related(normalized):
        return {"in_scope": True, "reason": None}
    if any(marker in normalized for marker in CANCER_MARKERS):
        return {"in_scope": False, "reason": "unrelated_cancer"}
    return {"in_scope": False, "reason": "breast_cancer_only"}


def filter_breast_cancer_evidence(results):
    return [
        result for result in results
        if is_breast_cancer_related(result.get("text", ""), result.get("source", ""))
    ]
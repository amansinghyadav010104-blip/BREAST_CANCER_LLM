import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    important medical information.
    """

    # 1. Normalize special characters
    text = text.replace("￾", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\xa0", " ")

    # 2. Join words split across PDF lines
    # Example: treat-\nment -> treatment
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # 3. Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # 4. Fix known PDF word-joining problems
    fixes = {
        "suchaschemotherapy": "such as chemotherapy",
        "suchas": "such as",
        "foreach": "for each",
        "partofyour": "part of your",
        "partof": "part of",
        "commonside": "common side",
        "optionsand": "options and",
        "mayinclude": "may include",
        "inthemilk": "in the milk",
        "inthe": "in the",
        "onthe": "on the",
        "ofthe": "of the",
        "forthe": "for the",
        "fromthe": "from the",
        "withthe": "with the",
        "andthe": "and the",
        "tothe": "to the",
        "aboutthe": "about the",
        "yourcancer": "your cancer",
        "cancertreatment": "cancer treatment",
        "breastcancer": "breast cancer",
        "cancercells": "cancer cells",
        "treatmentoptions": "treatment options",
        "treatmentplan": "treatment plan",
        "treatmentfor": "treatment for",
        "treatmentapproaches": "treatment approaches",
        "typesof": "types of",
        "typeof": "type of",
        "caninclude": "can include",
        "sideeffects": "side effects",
        "radiationtherapy": "radiation therapy",
        "hormonetherapy": "hormone therapy",
        "targetedtherapy": "targeted therapy",
        "triple-negativebreast": "triple-negative breast",
        "ductalcarcinoma": "ductal carcinoma",
        "inflammatorybreast": "inflammatory breast",
        "lumpectomyand": "lumpectomy and",
        "willwork": "will work",
        "treatment,which": "treatment, which",
        "differenttreatment": "different treatment",    
        "stageof": "stage of",
        "for specific": "for specific",
        " Ireceive": " I receive",
        "aclinical": "a clinical",
        "peoplewith": "people with",
        "clinicaltrial": "clinical trial",
        "currentNCI": "current NCI",
        "Noteveryone": "Not everyone",
        "Hormonetherapy": "Hormone therapy",
        "breast.Learn": "breast. Learn",
        "thedifferent": "the different",
        "BreastCancerTreatment": "Breast Cancer Treatment",
        "cancerTreatment": "cancer treatment",
        "breastcancer": "breast cancer",
        "cancerand": "cancer and",
        "cancerare": "cancer are",
        "cancerous": "cancerous",
        "to treat": "to treat",
    }

    for old, new in fixes.items():
        text = re.sub(
            re.escape(old),
            new,
            text,
            flags=re.IGNORECASE
        )

    # 5. Remove NCI timestamp/footer
    text = re.sub(
        r"\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}\s+"
        r"Treatment for Breast Cancer - NCI",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 6. Remove PDF page numbers
    text = re.sub(
        r"^\s*\d+/\d+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # 7. Normalize spaces and tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # 8. Remove whitespace before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    # 9. Clean individual lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # 10. Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()
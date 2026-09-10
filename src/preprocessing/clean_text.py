import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    important medical information.
    """

    # -----------------------------------------
    # 1. Replace unusual PDF characters
    # -----------------------------------------
    text = text.replace("￾", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")


    # -----------------------------------------
    # 2. Fix words broken across lines
    # Example:
    # treat-
    # ment
    # becomes:
    # treatment
    # -----------------------------------------
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )


    # -----------------------------------------
    # 3. Remove NCI PDF footer
    # -----------------------------------------
    text = re.sub(
        r"\d{2}/\d{2}/\d{4},.*?\n",
        "",
        text
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )


    # -----------------------------------------
    # 4. Fix words joined together
    # Example:
    # recurrenceof → recurrence of
    # inthe → in the
    # -----------------------------------------
    replacements = {
        "recurrenceof": "recurrence of",
        "triple-negativebreast": "triple-negative breast",
        "inthemilk": "in the milk",
        "aboutthe": "about the",
        "forthe": "for the",
        "ofthe": "of the",
        "withthe": "with the",
        "andthe": "and the",
        "inthe": "in the",
        "onthe": "on the",
        "tothe": "to the",
        "fromthe": "from the",
        "mayinclude": "may include",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)


    # -----------------------------------------
    # 5. Remove excessive spaces
    # -----------------------------------------
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # -----------------------------------------
    # 6. Remove spaces at beginning/end
    # -----------------------------------------
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )


    # -----------------------------------------
    # 7. Remove excessive blank lines
    # -----------------------------------------
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    # -----------------------------------------
    # 8. Remove spaces before punctuation
    # -----------------------------------------
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )


    return text.strip()


if __name__ == "__main__":

    sample_text = """
    Breast   Cancer     Treatment

    recurrenceof certain types of breast cancer.

    triple-negativebreast cancer.

    inthemilk ducts.

    29/08/2026, 21:36 Treatment for Breast Cancer - NCI
    https://www.cancer.gov/types/breast/treatment

    """

    cleaned = clean_text(sample_text)

    print("CLEANED TEXT")
    print("=" * 60)
    print(cleaned)
from enum import Enum


class AnswerLanguage(str, Enum):
    ENGLISH = "English only"
    HINDI = "Hindi only"
    BILINGUAL = "English + Hindi"


class AnswerLanguageGenerationError(RuntimeError):
    """Raised when Gemini's response does not honor the selected language."""


def get_language_instructions(language):
    try:
        language = AnswerLanguage(language)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported answer language: {language!r}") from error

    instructions = {
        AnswerLanguage.ENGLISH: "Write the complete answer in English only.",
        AnswerLanguage.HINDI: (
            "Write the complete answer in simple, natural Hindi (हिंदी) using Devanagari. "
            "Translate all headings and explanatory prose into Hindi. Do not return "
            "an English answer. Keep exact source filenames and [Evidence N] IDs."
        ),
        AnswerLanguage.BILINGUAL: (
            "Return two complete versions: first an English section headed exactly "
            "'## English', then a simple, natural Hindi section headed exactly "
            "'## हिंदी'. हिंदी अनुभाग अंग्रेज़ी उत्तर का पूरा अनुवाद हो। Translate every "
            "answer, evidence, and safety statement into "
            "Hindi in the second section. Reuse the exact same [Evidence N] IDs in "
            "both versions; keep exact source filenames."
        ),
    }
    return instructions[language]


def validate_answer_language(answer, language):
    language = AnswerLanguage(language)
    devanagari_count = sum("\u0900" <= char <= "\u097f" for char in answer)
    latin_count = sum(char.isascii() and char.isalpha() for char in answer)

    if language == AnswerLanguage.HINDI:
        forbidden_headings = ("### Answer", "### Evidence", "### Sources", "### Medical Safety")
        if devanagari_count < 12 or any(heading in answer for heading in forbidden_headings):
            raise AnswerLanguageGenerationError(
                "Gemini did not return a Hindi answer. Please retry or select another language."
            )
    elif language == AnswerLanguage.BILINGUAL:
        if (
            "## English" not in answer
            or "## हिंदी" not in answer
            or devanagari_count < 12
            or latin_count < 12
        ):
            raise AnswerLanguageGenerationError(
                "Gemini did not return both English and Hindi sections. Please retry."
            )
    elif devanagari_count > max(latin_count, 12):
        raise AnswerLanguageGenerationError(
            "Gemini did not return an English-only answer. Please retry."
        )
    return answer
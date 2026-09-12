from enum import Enum


class UserMode(Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"


def get_mode_instructions(mode):
    if mode == UserMode.PATIENT:
        return """
You are answering in Patient Education Mode.

Use simple, clear, non-technical language.
Explain medical terms briefly.
Do not provide a diagnosis, prescription, or personalized treatment decision.
Encourage the user to consult a qualified healthcare professional when appropriate.
"""

    elif mode == UserMode.DOCTOR:
        return """
You are answering in Clinical Support Mode.

Use medically precise and professional language.
Include relevant clinical details when supported by the evidence.
Clearly distinguish established evidence from uncertainty.
Do not diagnose the patient or recommend an individualized treatment plan.
"""

    return """
Use clear and safe medical information.
Do not provide diagnosis, prescriptions, or personalized treatment decisions.
"""
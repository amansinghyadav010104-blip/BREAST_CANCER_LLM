import streamlit as st


_SPEECH_COMPONENT = st.components.v2.component(
    "breast_cancer_speech_controls",
    html="""
<div>
  <button id="listen" type="button">🎙️ Speak question</button>
  <button id="play" type="button">🔊 Play answer</button>
  <span id="status" role="status" aria-live="polite"></span>
</div>
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const listen = parentElement.querySelector("#listen")
  const play = parentElement.querySelector("#play")
  const status = parentElement.querySelector("#status")
  if (!listen || !play || !status) return

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
  listen.onclick = () => {
    if (!SpeechRecognition) {
      status.textContent = "Speech recognition is unavailable in this browser."
      return
    }
    const recognition = new SpeechRecognition()
    recognition.lang = data.voice_locale
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.onstart = () => { status.textContent = "Listening..." }
    recognition.onerror = (event) => {
      status.textContent = event.error === "not-allowed"
        ? "Allow microphone access to use voice input."
        : "Speech recognition failed. You can type your question instead."
    }
    recognition.onend = () => {
      if (status.textContent === "Listening...") status.textContent = ""
    }
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript?.trim()
      if (transcript) {
        setTriggerValue("transcript", transcript)
        status.textContent = "Question transcribed."
      }
    }
    try {
      recognition.start()
    } catch (_) {
      status.textContent = "Could not start the microphone. Try again or type your question."
    }
  }

  play.disabled = !data.answer_text
  play.onclick = () => {
    if (!data.answer_text || !window.speechSynthesis) {
      status.textContent = "Speech playback is unavailable in this browser."
      return
    }
    window.speechSynthesis.cancel()
    const answer = data.answer_text.replace(/[#*_`]/g, "")
    const parts = answer.split(/([\u0900-\u097F]+)/).filter(Boolean)
    for (const part of parts) {
      const utterance = new SpeechSynthesisUtterance(part)
      utterance.lang = /[\u0900-\u097F]/.test(part) ? "hi-IN" : "en-IN"
      window.speechSynthesis.speak(utterance)
    }
    status.textContent = "Playing answer."
  }

  return () => {
    listen.onclick = null
    play.onclick = null
  }
}
""",
)


def speech_controls(voice_locale, answer_text, key="breast-cancer-speech"):
    result = _SPEECH_COMPONENT(
        key=key,
        data={"voice_locale": voice_locale, "answer_text": answer_text or ""},
        on_transcript_change=lambda: None,
    )
    return getattr(result, "transcript", None)
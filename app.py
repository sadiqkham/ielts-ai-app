import io
import re
import streamlit as st
from groq import Groq
from gtts import gTTS

# ================= SETTINGS =================
LLM_MODEL = "llama-3.3-70b-versatile"
STT_MODEL = "whisper-large-v3-turbo"

CORRECTION_PROMPT = """You are an expert English teacher helping a learner prepare for IELTS.
The learner may have written or spoken the sentence, so ignore minor punctuation/capitalisation from speech-to-text.

Analyse the learner's message and reply in EXACTLY this format (Markdown):

### Original
> (the learner's sentence)

### Corrected
> (grammatically correct version)

### Mistakes
- "wrong part" ❌ → "right part" ✅
(If there are no mistakes, write: No mistakes. Well done!)

### Simple Explanation
(Short, easy explanation of the rules, with one example)

### Natural English
> (a more natural / higher-band way to say the same thing)

### Conversation
(Reply naturally to what the learner said, then ask ONE follow-up question to continue the conversation.)
"""

IELTS_EXAMINER_PROMPT = """You are a friendly but professional IELTS Speaking examiner.
Conduct the test in three parts, one question at a time:
- Part 1: personal questions (hometown, studies, hobbies, daily life). Ask about 4 questions.
- Part 2: give a cue card ("Describe ...") with 3-4 bullet points, then wait for a 1-2 minute answer.
- Part 3: 3-4 deeper analytical questions linked to the Part 2 topic.

Rules:
- Ask only ONE question at a time and wait for the answer.
- Keep your own turns short.
- Briefly acknowledge answers (e.g. "Thank you.") without long feedback during the test.
- Tell the learner clearly when you move to the next part.
- Do NOT give grading during the test unless asked for the final feedback.
"""

IELTS_FEEDBACK_PROMPT = """You are an IELTS Speaking examiner. Below is a transcript of a practice session.
Give feedback in this format (Markdown):

| Category | Rating | Feedback |
|---|---|---|
| Grammar | Good / Needs Improvement | ... |
| Vocabulary | ... | ... |
| Fluency | ... | ... |
| Sentence Structure | ... | ... |
| Pronunciation | ... | ... |
| Coherence | ... | ... |

Then give:
- **Top 3 mistakes** the learner repeated (with corrections)
- **3 tips** to improve
- **Estimated Speaking Band: X.X**

IMPORTANT: Clearly say this is an AI estimate based on a text transcript, NOT an official IELTS score.
Pronunciation cannot be judged reliably from text, so say that too.
"""

# ================= AI FUNCTIONS =================
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def chat(messages, temperature=0.4):
    response = get_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature
    )
    return response.choices[0].message.content

def correct_and_reply(history, user_text, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-10:]
    messages.append({"role": "user", "content": user_text})
    return chat(messages)

def transcribe(audio_bytes):
    result = get_client().audio.transcriptions.create(
        file=("audio.wav", audio_bytes),
        model=STT_MODEL,
        language="en",
        response_format="text",
    )
    return str(result).strip()

def speak(text, max_chars=800):
    text = re.sub(r"[#*>`_|]", "", text)
    text = re.sub(r"[❌✅]", "", text)
    text = re.sub(r"\s+", " ", text).strip()[:max_chars]
    if not text:
        return None
    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    return buf.getvalue()

# ================= PAGE =================
st.set_page_config(page_title="IELTS English AI", page_icon="🎓", layout="wide")
st.title("🎓 IELTS English Speaking & Grammar AI")

mode = st.sidebar.radio("Mode", ["📝 Grammar & Conversation", "🎤 IELTS Speaking Test"])
use_tts = st.sidebar.checkbox("🔊 Read AI reply aloud", value=True)

if st.sidebar.button("🗑️ Reset conversation"):
    for k in ["grammar_chat", "ielts_chat", "feedback", "pending_voice"]:
        st.session_state.pop(k, None)
    st.rerun()

st.session_state.setdefault("grammar_chat", [])
st.session_state.setdefault("ielts_chat", [])
st.session_state.setdefault("mic_n", 0)

def play_voice(text):
    try:
        audio_bytes = speak(text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Voice failed: {e}")

def play_pending_voice():
    text = st.session_state.pop("pending_voice", None)
    if use_tts and text:
        play_voice(text)

def get_user_input(key):
    st.write("**Speak or type your message:**")
    audio = st.audio_input("🎤 Click to record", key=f"mic_{key}_{st.session_state.mic_n}")
    typed = st.chat_input("Or type here...", key=f"txt_{key}")
    if audio is not None:
        st.session_state.mic_n += 1
        try:
            with st.spinner("Transcribing your voice..."):
                return transcribe(audio.getvalue())
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")
            return None
    return typed

def show_history(chat_list):
    for m in chat_list:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ================= MODE 1 =================
if mode.startswith("📝"):
    st.subheader("Grammar Correction + Conversation")
    show_history(st.session_state.grammar_chat)
    play_pending_voice()
    user_text = get_user_input("grammar")

    if user_text:
        with st.spinner("Checking your English..."):
            reply = correct_and_reply(st.session_state.grammar_chat, user_text, CORRECTION_PROMPT)
        st.session_state.grammar_chat.append({"role": "user", "content": user_text})
        st.session_state.grammar_chat.append({"role": "assistant", "content": reply})
        st.session_state.pending_voice = reply.split("### Conversation")[-1]
        st.rerun()

# ================= MODE 2 =================
else:
    st.subheader("IELTS Speaking Test (AI Examiner)")
    st.caption("Feedback is an AI estimate, not an official IELTS score.")

    if not st.session_state.ielts_chat:
        if st.button("▶️ Start Test (Part 1)"):
            first = chat([
                {"role": "system", "content": IELTS_EXAMINER_PROMPT},
                {"role": "user", "content": "Please start the test with a short greeting and the first Part 1 question."},
            ])
            st.session_state.ielts_chat.append({"role": "assistant", "content": first})
            st.session_state.pending_voice = first
            st.rerun()
    else:
        show_history(st.session_state.ielts_chat)
        play_pending_voice()
        user_text = get_user_input("ielts")

        if user_text:
            st.session_state.ielts_chat.append({"role": "user", "content": user_text})
            with st.spinner("Examiner is thinking..."):
                msgs = [{"role": "system", "content": IELTS_EXAMINER_PROMPT}]
                msgs += st.session_state.ielts_chat
                reply = chat(msgs)
            st.session_state.ielts_chat.append({"role": "assistant", "content": reply})
            st.session_state.pending_voice = reply
            st.rerun()

        if st.button("📊 Finish & Get Feedback"):
            transcript = "\n".join(
                f"{'Examiner' if m['role']=='assistant' else 'Candidate'}: {m['content']}"
                for m in st.session_state.ielts_chat
            )
            with st.spinner("Analysing your session..."):
                st.session_state.feedback = chat([
                    {"role": "system", "content": IELTS_FEEDBACK_PROMPT},
                    {"role": "user", "content": transcript},
                ], temperature=0.2)

        if st.session_state.get("feedback"):
            st.markdown("---")
            st.markdown("## 📊 Your IELTS Feedback")
            st.markdown(st.session_state.feedback)

"""Notes & Quiz Generator: upload a chapter/book, get pro notes + an interactive quiz."""
import io
import json
import time

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-2.5-flash"
CHUNK_CHARS = 30_000       # size of each piece sent for notes
MAX_CHUNKS = 25            # safety cap for very large books
QUIZ_DIRECT_LIMIT = 150_000  # above this, the quiz is built from the notes instead


class Question(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str


# ---------- File handling ----------
def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")  # txt / md


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries so chunks stay readable."""
    chunks, current = [], ""
    for para in text.split("\n"):
        if len(current) + len(para) + 1 > size and current:
            chunks.append(current)
            current = ""
        while len(para) > size:  # a single huge paragraph
            chunks.append(para[:size])
            para = para[size:]
        current += para + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


# ---------- Gemini ----------
def call_gemini(client, model, prompt, system, schema=None, retries=3):
    cfg = {"system_instruction": system, "temperature": 0.4}
    if schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = schema
    last = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
            if not resp.text:
                raise ValueError("Empty response (possibly blocked by safety filters).")
            return resp.text
        except Exception as e:  # rate limits / transient errors
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request failed: {last}")


NOTES_SYSTEM = (
    "You are an expert tutor who writes professional, accurate study notes. "
    "Use only the provided text. Never invent facts."
)


def generate_notes(client, model, chunks, style, progress=None):
    parts = []
    for i, ch in enumerate(chunks, 1):
        prompt = (
            f"Create {style} study notes in Markdown from the text below.\n"
            "Structure: ## Section headings, bullet points, **bold** key terms, "
            "a short 'Key Takeaways' list at the end, and a 'Key Terms' glossary if relevant.\n\n"
            f"TEXT (part {i} of {len(chunks)}):\n{ch}"
        )
        parts.append(call_gemini(client, model, prompt, NOTES_SYSTEM))
        if progress:
            progress(i / len(chunks))
    return "\n\n---\n\n".join(parts)


def validate_questions(raw: str) -> list[dict]:
    items = json.loads(raw)
    good = []
    for q in items:
        try:
            m = Question(**q)
        except Exception:
            continue
        if len(m.options) >= 2 and 0 <= m.answer_index < len(m.options):
            good.append(m.model_dump())
    return good


def generate_quiz(client, model, source, n, difficulty):
    prompt = (
        f"Write exactly {n} multiple-choice questions ({difficulty} difficulty) from the material below. "
        "Each question has exactly 4 options, one correct answer (answer_index is 0-based), "
        "and a short explanation. Cover the whole material, avoid trivial or duplicate questions, "
        "and do not use 'all of the above'.\n\nMATERIAL:\n" + source
    )
    raw = call_gemini(client, model, prompt, NOTES_SYSTEM, schema=list[Question])
    return validate_questions(raw)


# ---------- UI ----------
def main():
    st.set_page_config(page_title="Notes & Quiz Generator", page_icon="📚", layout="wide")
    st.title("📚 Notes & Quiz Generator")
    st.caption("Upload a chapter or book → get professional notes and a practice quiz.")

    with st.sidebar:
        st.header("Settings")
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Get one free at aistudio.google.com")
        model = st.text_input("Model", DEFAULT_MODEL)
        style = st.selectbox("Notes style", ["concise", "detailed", "exam-focused"], index=1)
        n_q = st.slider("Number of quiz questions", 5, 30, 10)
        diff = st.selectbox("Quiz difficulty", ["easy", "medium", "hard"], index=1)

    uploaded = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"])
    pasted = st.text_area("...or paste text", height=120)

    if st.button("Generate", type="primary"):
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar.")
            st.stop()
        try:
            text = extract_text(uploaded) if uploaded else pasted
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        if len(text.strip()) < 200:
            st.error("Not enough text found. Scanned PDFs (images) are not supported.")
            st.stop()

        client = genai.Client(api_key=api_key)
        chunks = chunk_text(text)
        if len(chunks) > MAX_CHUNKS:
            st.warning(f"Large document: only the first {MAX_CHUNKS} parts will be processed.")
            chunks = chunks[:MAX_CHUNKS]
        try:
            bar = st.progress(0.0, text="Writing notes...")
            notes = generate_notes(client, model, chunks, style,
                                   lambda p: bar.progress(p, text="Writing notes..."))
            bar.progress(1.0, text="Building quiz...")
            source = "\n".join(chunks)
            if len(source) > QUIZ_DIRECT_LIMIT:
                source = notes
            quiz = generate_quiz(client, model, source, n_q, diff)
            bar.empty()
        except Exception as e:
            st.error(str(e))
            st.stop()
        st.session_state.update(notes=notes, quiz=quiz, submitted=False,
                                qid=st.session_state.get("qid", 0) + 1)

    if "notes" not in st.session_state:
        return

    tab_notes, tab_quiz = st.tabs(["📝 Notes", "❓ Quiz"])
    with tab_notes:
        st.markdown(st.session_state["notes"])
        st.download_button("Download notes (.md)", st.session_state["notes"], "notes.md")

    with tab_quiz:
        quiz = st.session_state["quiz"]
        if not quiz:
            st.warning("No valid questions were generated. Try again.")
            return
        with st.form("quiz_form"):
            answers = []
            for i, q in enumerate(quiz):
                answers.append(st.radio(f"**Q{i+1}. {q['question']}**", q["options"], index=None,
                                        key=f"q{st.session_state['qid']}_{i}"))
            submit = st.form_submit_button("Submit answers")
        if submit:
            st.session_state["submitted"] = True
        if st.session_state.get("submitted"):
            score = 0
            for i, q in enumerate(quiz):
                correct = q["options"][q["answer_index"]]
                ok = answers[i] == correct
                score += ok
                (st.success if ok else st.error)(
                    f"Q{i+1}: {'Correct' if ok else 'Incorrect'}. Answer: {correct}\n\n{q['explanation']}")
            st.metric("Score", f"{score} / {len(quiz)}")
        st.download_button("Download quiz (.json)", json.dumps(quiz, indent=2), "quiz.json")


if __name__ == "__main__":
    main()

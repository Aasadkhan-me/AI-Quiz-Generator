
"""Notes & Quiz Generator: upload a chapter/book, get professional notes and an interactive quiz."""

import io
import json
import time

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pydantic import BaseModel
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-3.8-flash"
CHUNK_CHARS = 30_000
MAX_CHUNKS = 25
QUIZ_DIRECT_LIMIT = 150_000


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
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    chunks = []
    current = ""

    for para in text.split("\n"):
        if len(current) + len(para) + 1 > size and current:
            chunks.append(current)
            current = ""

        while len(para) > size:
            chunks.append(para[:size])
            para = para[size:]

        current += para + "\n"

    if current.strip():
        chunks.append(current)

    return chunks


# ---------- Gemini API ----------

def call_gemini(client, model, prompt, system, schema=None, retries=3):
    config = {
        "system_instruction": system,
        "temperature": 0.4,
    }

    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema

    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )

            if not response.text:
                raise ValueError(
                    "Empty response. The request may have been blocked "
                    "or returned no text."
                )

            return response.text

        except Exception as error:
            last_error = error
            message = str(error)
            upper_message = message.upper()

            if (
                "NOT_FOUND" in upper_message
                or "404" in upper_message
            ):
                raise RuntimeError(
                    f"Model '{model}' was not found or is unavailable. "
                    "Check the model name and API-key access. "
                    f"Original error: {message}"
                ) from error

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gemini request failed after {retries} attempts: {last_error}"
    )


NOTES_SYSTEM = (
    "You are an expert tutor who creates accurate, professional study notes. "
    "Use only the provided source text. Do not invent facts."
)


# ---------- Notes generation ----------

def generate_notes(client, model, chunks, style, progress=None):
    parts = []

    for i, chunk in enumerate(chunks, 1):
        prompt = (
            f"Create {style} study notes in Markdown from the text below.\n"
            "Use clear section headings, bullet points, bold key terms, "
            "a Key Takeaways list, and a Key Terms glossary where relevant.\n\n"
            f"TEXT (part {i} of {len(chunks)}):\n{chunk}"
        )

        notes_part = call_gemini(
            client, model, prompt, NOTES_SYSTEM
        )
        parts.append(notes_part)

        if progress:
            progress(i / len(chunks))

    return "\n\n---\n\n".join(parts)


# ---------- Quiz generation ----------

def validate_questions(raw: str) -> list[dict]:
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(items, list):
        return []

    good = []

    for item in items:
        try:
            question = Question(**item)
        except Exception:
            continue

        if (
            len(question.options) == 4
            and 0 <= question.answer_index < len(question.options)
        ):
            good.append(question.model_dump())

    return good


def generate_quiz(client, model, source, number, difficulty):
    prompt = (
        f"Create exactly {number} multiple-choice questions at "
        f"{difficulty} difficulty from the material below. "
        "Each question must have exactly four options, one correct answer, "
        "a zero-based answer_index, and a short explanation. "
        "Cover the material broadly, avoid duplicate questions, and do not "
        "use 'all of the above'. Return a JSON array.\n\n"
        f"MATERIAL:\n{source}"
    )

    raw = call_gemini(
        client,
        model,
        prompt,
        NOTES_SYSTEM,
        schema=list[Question],
    )

    return validate_questions(raw)


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(
        page_title="Notes & Quiz Generator",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Notes & Quiz Generator")
    st.caption(
        "Upload a chapter or book to generate professional study notes "
        "and an interactive practice quiz."
    )

    with st.sidebar:
        st.header("Settings")

        secret_key = st.secrets.get("GEMINI_API_KEY", "")

        api_key = st.text_input(
            "Gemini API key",
            value=secret_key,
            type="password",
            help="Store your key in Streamlit secrets for deployment.",
        )

        model = st.text_input("Gemini model", DEFAULT_MODEL)

        style = st.selectbox(
            "Notes style",
            ["concise", "detailed", "exam-focused"],
            index=1,
        )

        number = st.slider(
            "Number of quiz questions", 5, 30, 10
        )

        difficulty = st.selectbox(
            "Quiz difficulty",
            ["easy", "medium", "hard"],
            index=1,
        )

    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown",
        type=["pdf", "docx", "txt", "md"],
    )

    pasted = st.text_area(
        "Or paste your chapter text",
        height=120,
    )

    if st.button("Generate Notes & Quiz", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key.")
            st.stop()

        if not uploaded and not pasted.strip():
            st.error("Upload a file or paste your text first.")
            st.stop()

        try:
            source_text = (
                extract_text(uploaded) if uploaded else pasted
            )
        except Exception as error:
            st.error(f"Could not read the file: {error}")
            st.stop()

        if len(source_text.strip()) < 200:
            st.error(
                "Not enough text found. Scanned PDFs without extractable "
                "text are not supported."
            )
            st.stop()

        chunks = chunk_text(source_text)

        if len(chunks) > MAX_CHUNKS:
            st.warning(
                f"Large document: processing only the first "
                f"{MAX_CHUNKS} sections."
            )
            chunks = chunks[:MAX_CHUNKS]

        client = genai.Client(api_key=api_key)
        progress = st.progress(0.0, text="Generating study notes...")

        try:
            notes = generate_notes(
                client,
                model,
                chunks,
                style,
                lambda value: progress.progress(
                    value, text="Generating study notes..."
                ),
            )

            progress.progress(1.0, text="Creating your quiz...")

            quiz_source = "\n\n".join(chunks)

            if len(quiz_source) > QUIZ_DIRECT_LIMIT:
                quiz_source = notes

            quiz = generate_quiz(
                client,
                model,
                quiz_source,
                number,
                difficulty,
            )

            st.session_state["notes"] = notes
            st.session_state["quiz"] = quiz
            st.session_state["qid"] = (
                st.session_state.get("qid", 0) + 1
            )
            st.session_state["submitted"] = False

        except Exception as error:
            st.error(str(error))
            st.stop()

        finally:
            progress.empty()

    if "notes" not in st.session_state:
        return

    notes_tab, quiz_tab = st.tabs(["📝 Study Notes", "❓ Practice Quiz"])

    with notes_tab:
        st.markdown(st.session_state["notes"])

        st.download_button(
            "Download notes (.md)",
            data=st.session_state["notes"],
            file_name="study_notes.md",
            mime="text/markdown",
        )

    with quiz_tab:
        quiz = st.session_state["quiz"]

        if not quiz:
            st.warning(
                "No valid questions were generated. Please try again."
            )
            return

        with st.form(
            f"quiz_form_{st.session_state['qid']}"
        ):
            answers = []

            for i, question in enumerate(quiz):
                answer = st.radio(
                    f"Q{i + 1}. {question['question']}",
                    question["options"],
                    index=None,
                    key=f"q{st.session_state['qid']}_{i}",
                )
                answers.append(answer)

            submitted = st.form_submit_button("Submit answers")

        if submitted:
            st.session_state["submitted"] = True
            st.session_state["answers"] = answers

        if st.session_state.get("submitted"):
            saved_answers = st.session_state["answers"]
            score = 0

            for i, question in enumerate(quiz):
                correct = question["options"][
                    question["answer_index"]
                ]
                selected = saved_answers[i]
                is_correct = selected == correct

                if is_correct:
                    score += 1
                    st.success(f"Q{i + 1}: Correct!")
                else:
                    st.error(f"Q{i + 1}: Incorrect.")

                if selected is None:
                    st.write("You did not answer this question.")

                st.write(f"Correct answer: {correct}")
                st.write(f"Explanation: {question['explanation']}")

            st.metric("Your score", f"{score} / {len(quiz)}")

        st.download_button(
            "Download quiz (.json)",
            data=json.dumps(quiz, indent=2, ensure_ascii=False),
            file_name="practice_quiz.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()



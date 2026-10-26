# 📚 Notes & Quiz Generator

Upload a chapter or a whole book (PDF, DOCX, TXT, MD) and get professional study notes plus an interactive multiple-choice quiz, powered by Google Gemini Flash and Streamlit.

## Features
- Notes in 3 styles: concise, detailed, exam-focused
- Large books are split into parts and processed automatically
- Quiz with 5–30 questions, difficulty levels, scoring and explanations
- Download notes (.md) and quiz (.json)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Get a free API key at https://aistudio.google.com and paste it in the sidebar, or create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key"
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add `GEMINI_API_KEY = "your-key"`.
5. Click **Deploy**.

## Notes
- Scanned (image-only) PDFs are not supported; they need OCR first.
- Change the model name in the sidebar if Google releases a newer Flash model.
- Never commit your API key.

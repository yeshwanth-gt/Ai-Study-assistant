import streamlit as st
import pdfplumber
import docx
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import random

# Download NLTK resources
nltk.download('punkt')

# -----------------------------
# Document Parsing Functions
# -----------------------------
def parse_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def parse_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def parse_document(file_path):
    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".docx"):
        return parse_docx(file_path)
    elif file_path.endswith(".txt"):
        return parse_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

# -----------------------------
# Summarization (Sumy LSA)
# -----------------------------
def generate_summary(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

# -----------------------------
# Q&A Engine (Keyword-based)
# -----------------------------
def answer_question(text, question):
    sentences = sent_tokenize(text)
    question_words = question.lower().split()
    scored = []
    for sent in sentences:
        score = sum(1 for word in question_words if word in sent.lower())
        if score > 0:
            scored.append((score, sent))
    if not scored:
        return "Sorry, I couldn't find an answer in the document."
    scored.sort(reverse=True)
    best_sentences = [s for _, s in scored[:2]]
    return " ".join(best_sentences)

# -----------------------------
# Flashcards Generator
# -----------------------------
def generate_flashcards(text, max_cards=5):
    sentences = sent_tokenize(text)
    flashcards = []
    for sent in sentences:
        if " is " in sent.lower():
            parts = sent.split(" is ")
            if len(parts) == 2:
                question = f"What is {parts[0].strip()}?"
                answer = parts[1].strip()
                flashcards.append((question, answer))
        if len(flashcards) >= max_cards:
            break
    if not flashcards:
        return [("No flashcards generated", "Try uploading a document with definitions.")]
    return flashcards

# -----------------------------
# Quiz Generator (MCQs)
# -----------------------------
def generate_quiz(text, num_questions=5):
    sentences = sent_tokenize(text)
    quiz = []
    for i in range(min(num_questions, len(sentences))):
        sent = random.choice(sentences)
        words = sent.split()
        if len(words) > 5:
            correct_answer = words[1]
            options = random.sample(words, min(4, len(words)))
            if correct_answer not in options:
                options[0] = correct_answer
            random.shuffle(options)
            quiz.append((sent, correct_answer, options))
    return quiz

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("📚 AI Study Assistant")
st.write("Upload your study material and let AI help you summarize, answer questions, create flashcards, and practice quizzes.")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        text = parse_document(uploaded_file.name)
        st.success("✅ Document uploaded and parsed successfully!")

        # Preview
        with st.expander("📄 Document Preview"):
            st.text_area("Extracted Text", text[:2000], height=300)

        # Summarization
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = generate_summary(text)
            st.subheader("📝 Summary")
            st.write(summary)

        # Q&A
        st.subheader("❓ Ask Questions")
        user_question = st.text_input("Enter your question about the document:")
        if user_question:
            with st.spinner("Searching for answer..."):
                answer = answer_question(text, user_question)
            st.subheader("💡 Answer")
            st.write(answer)

        # Flashcards
        st.subheader("🎴 Flashcards")
        if st.button("Generate Flashcards"):
            with st.spinner("Creating flashcards..."):
                flashcards = generate_flashcards(text)
            for i, (q, a) in enumerate(flashcards, 1):
                st.markdown(f"**Q{i}: {q}**")
                st.write(f"➡️ {a}")

        # Quiz Mode
        st.subheader("🧩 Quiz Mode")
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz..."):
                quiz = generate_quiz(text)
            for i, (sent, correct, options) in enumerate(quiz, 1):
                st.markdown(f"**Q{i}: Based on this sentence:**")
                st.write(f"➡️ {sent}")
                st.write("Options:")
                for opt in options:
                    if opt == correct:
                        st.write(f"- ✅ {opt}")
                    else:
                        st.write(f"- {opt}")

    except Exception as e:
        st.error(f"Error parsing document: {e}")
else:
    st.warning("Please upload a document to get started.")
"""
EDU-02: Flashcard Generator from Notes
----------------------------------------
A RAG-powered Streamlit app that lets a user upload their notes (PDF),
automatically generates flashcards from the content, and offers a
revision / quiz mode to test recall.

Pipeline:
    PDF Upload -> PyPDFLoader -> Text Chunking -> OpenAI Embeddings
    -> FAISS Vector Store -> LLM Flashcard Generation -> Flashcard / Quiz UI
"""

import os
import json
import random
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document


# ----------------------------- Page Config ----------------------------- #
st.set_page_config(page_title="Flashcard Generator from Notes", page_icon="🗂️", layout="wide")


# ----------------------------- Session State ---------------------------- #
def init_state():
    defaults = {
        "vectorstore": None,
        "chunks": [],
        "flashcards": [],
        "card_index": 0,
        "show_answer": False,
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_answers": {},
        "quiz_submitted": False,
        "processed_filename": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ------------------------------- Sidebar -------------------------------- #
with st.sidebar:
    st.title("🗂️ Flashcard Generator")
    st.caption("RAG over your uploaded notes")

    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    uploaded_file = st.file_uploader("Upload your notes (PDF)", type=["pdf"])

    num_cards = st.slider("Number of flashcards to generate", min_value=5, max_value=30, value=10, step=5)
    chunk_size = st.slider("Chunk size (characters)", min_value=500, max_value=2000, value=1000, step=100)

    process_btn = st.button("Process Notes", type="primary", use_container_width=True)

    st.divider()
    st.caption("Tech stack: Streamlit · LangChain · FAISS · OpenAI")


# --------------------------- Core RAG Pipeline --------------------------- #
def load_and_chunk_pdf(uploaded_file, chunk_size=1000, chunk_overlap=150):
    """Save the uploaded PDF to a temp file, load it, and split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    os.unlink(tmp_path)
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and build a FAISS vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def generate_flashcards(chunks, num_cards=10):
    """Use an LLM to turn note chunks into structured Q&A flashcards."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Sample chunks so we cover the breadth of the notes without
    # blowing past context limits on very long documents.
    sample = chunks if len(chunks) <= 25 else random.sample(chunks, 25)
    combined_text = "\n\n---\n\n".join(c.page_content for c in sample)

    prompt = f"""You are an expert study-assistant. Read the notes below and create
{num_cards} high-quality flashcards for exam revision.

Rules:
- Each flashcard must have a concise "question" and a precise "answer".
- Cover distinct concepts; avoid duplicate or overlapping questions.
- Prefer questions that test understanding, not just recall of trivia.
- Return ONLY valid JSON: a list of objects with keys "question" and "answer".
  No markdown fences, no commentary.

NOTES:
{combined_text}
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Defensive cleanup in case the model wraps the JSON in a code fence.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        cards = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        cards = json.loads(raw[start:end + 1])

    return cards


def answer_from_notes(vectorstore, question, k=4):
    """RAG: retrieve relevant chunks and answer a question about the notes."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""Answer the question using ONLY the context from the user's notes below.
If the answer isn't in the notes, say so clearly.

CONTEXT:
{context}

QUESTION: {question}
"""
    response = llm.invoke(prompt)
    return response.content, docs


def grade_quiz_answer(question, correct_answer, user_answer):
    """Use the LLM as a lenient grader for free-text quiz answers."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Question: {question}
Correct answer: {correct_answer}
Student's answer: {user_answer}

Judge if the student's answer is substantially correct (minor wording differences are fine).
Respond ONLY with valid JSON: {{"correct": true/false, "feedback": "one short sentence"}}"""
    response = llm.invoke(prompt)
    raw = response.content.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"correct": False, "feedback": "Could not grade automatically."}


# ------------------------------ Processing ------------------------------- #
if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.sidebar.error("Please upload a PDF of your notes.")
    else:
        with st.spinner("Reading and chunking your notes..."):
            chunks = load_and_chunk_pdf(uploaded_file, chunk_size=chunk_size)
            st.session_state.chunks = chunks

        with st.spinner("Building FAISS vector index..."):
            st.session_state.vectorstore = build_vectorstore(chunks)

        with st.spinner("Generating flashcards..."):
            st.session_state.flashcards = generate_flashcards(chunks, num_cards=num_cards)

        st.session_state.processed_filename = uploaded_file.name
        st.session_state.card_index = 0
        st.session_state.show_answer = False
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False

        st.sidebar.success(f"Processed '{uploaded_file.name}' — {len(chunks)} chunks, "
                            f"{len(st.session_state.flashcards)} flashcards ready.")


# --------------------------------- Tabs ---------------------------------- #
tab_flash, tab_quiz, tab_chat = st.tabs(["🗂️ Flashcards", "📝 Quiz Mode", "💬 Ask Your Notes"])

# ----- Flashcards tab ----- #
with tab_flash:
    st.header("Flashcard Revision")

    if not st.session_state.flashcards:
        st.info("Upload a PDF of your notes and click **Process Notes** in the sidebar to begin.")
    else:
        cards = st.session_state.flashcards
        idx = st.session_state.card_index
        card = cards[idx]

        st.caption(f"Card {idx + 1} of {len(cards)}  ·  Source: {st.session_state.processed_filename}")

        card_html = f"""
        <div style="
            border: 1px solid #ddd; border-radius: 12px; padding: 32px;
            min-height: 160px; display:flex; align-items:center; justify-content:center;
            text-align:center; font-size:20px; background-color:#fafafa;">
            {card['answer'] if st.session_state.show_answer else card['question']}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        st.write("")

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("⬅️ Previous", use_container_width=True, disabled=idx == 0):
                st.session_state.card_index -= 1
                st.session_state.show_answer = False
                st.rerun()
        with col2:
            label = "Show Answer" if not st.session_state.show_answer else "Show Question"
            if st.button(f"🔄 {label}", use_container_width=True):
                st.session_state.show_answer = not st.session_state.show_answer
                st.rerun()
        with col3:
            if st.button("Next ➡️", use_container_width=True, disabled=idx == len(cards) - 1):
                st.session_state.card_index += 1
                st.session_state.show_answer = False
                st.rerun()
        with col4:
            if st.button("🔀 Shuffle", use_container_width=True):
                random.shuffle(cards)
                st.session_state.card_index = 0
                st.session_state.show_answer = False
                st.rerun()

        with st.expander("View all flashcards"):
            for i, c in enumerate(cards, start=1):
                st.markdown(f"**{i}. Q:** {c['question']}")
                st.markdown(f"**A:** {c['answer']}")
                st.divider()

# ----- Quiz Mode tab ----- #
with tab_quiz:
    st.header("Quiz Mode")

    if not st.session_state.flashcards:
        st.info("Process your notes first to unlock quiz mode.")
    else:
        cards = st.session_state.flashcards
        qidx = st.session_state.quiz_index

        if qidx < len(cards):
            card = cards[qidx]
            st.progress(qidx / len(cards))
            st.subheader(f"Question {qidx + 1} of {len(cards)}")
            st.write(card["question"])

            user_answer = st.text_input("Your answer", key=f"quiz_input_{qidx}")

            if st.button("Submit Answer", key=f"submit_{qidx}"):
                if user_answer.strip():
                    result = grade_quiz_answer(card["question"], card["answer"], user_answer)
                    st.session_state.quiz_answers[qidx] = {
                        "user_answer": user_answer,
                        "correct": result.get("correct", False),
                        "feedback": result.get("feedback", ""),
                        "correct_answer": card["answer"],
                    }
                    if result.get("correct"):
                        st.session_state.quiz_score += 1
                    st.session_state.quiz_index += 1
                    st.rerun()
                else:
                    st.warning("Type an answer before submitting.")
        else:
            st.success(f"Quiz complete! Score: {st.session_state.quiz_score} / {len(cards)}")
            with st.expander("Review your answers"):
                for i, res in st.session_state.quiz_answers.items():
                    icon = "✅" if res["correct"] else "❌"
                    st.markdown(f"{icon} **Q{i + 1}: {cards[i]['question']}**")
                    st.markdown(f"- Your answer: {res['user_answer']}")
                    st.markdown(f"- Correct answer: {res['correct_answer']}")
                    st.markdown(f"- Feedback: {res['feedback']}")
                    st.divider()

            if st.button("Retake Quiz"):
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = {}
                st.rerun()

# ----- Ask Your Notes (RAG chat) tab ----- #
with tab_chat:
    st.header("Ask Your Notes")
    st.caption("Ask a free-form question — answers are retrieved directly from your uploaded notes (RAG).")

    if not st.session_state.vectorstore:
        st.info("Process your notes first to enable Q&A.")
    else:
        question = st.text_input("Ask something about your notes")
        if st.button("Ask") and question.strip():
            with st.spinner("Searching your notes..."):
                answer, sources = answer_from_notes(st.session_state.vectorstore, question)
            st.markdown("**Answer:**")
            st.write(answer)
            with st.expander("Retrieved context"):
                for i, doc in enumerate(sources, start=1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} (page {page}):**")
                    st.write(doc.page_content)
                    st.divider()

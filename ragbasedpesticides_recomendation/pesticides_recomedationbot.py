import streamlit as st
import os
from typing import TypedDict, Annotated, List
import wikipedia

# LangChain & LangGraph imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
import google.generativeai as genai

# --- ⚙️ Configuration ---
GOOGLE_API_KEY = "AIzaSyDo0N65nL2pb2cqfyojrk0wb0a0osNDfEU"

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    st.error("🛑 Please set your GOOGLE_API_KEY before running.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Streamlit UI ---
st.set_page_config(page_title="Pesticides Recommendation", layout="wide")
st.title("🌾 Your AI Assistant ")


# Sidebar
with st.sidebar:
    
    pdf_files = [r"K:\testing\FarmSathi-AI-Powered-Farmers-Assistant\ragbasedpesticides_recomendation\Pesticides_recomendation_latest.pdf"]  # Replace or make upload option
    

# --- Retriever Setup ---
@st.cache_resource
def build_retriever(pdf_paths: List[str]):
    docs = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠️ Could not load PDF {path}: {e}")
    if not docs:
        st.warning("No PDFs loaded. Only Wikipedia fallback will work.")

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embedder)
        return db.as_retriever(search_kwargs={"k": 4})
    return None

retriever = build_retriever(pdf_files)

# --- State Definition ---
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str

# --- Main RAG Node ---
def rag_node(state: GraphState):
    messages = state["messages"]
    query = messages[-1].content

    context_str = ""
    if retriever:
        docs = retriever.invoke(query)
        context_str = "\n\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # --- Step 1: Try RAG from PDF ---
    prompt = (
        "You are an expert agricultural assistant. "
        "Use the PDF context below to answer accurately. "
        "Try to explain like Farmer Can understand easily. "
        "If no direct recommendation is found, say so truthfully.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as e:
        return {"messages": [AIMessage(content=f"⚠️ Gemini error: {e}")], "context": context_str}

    # --- Step 2: Check if the answer lacks pesticide information ---
    weak_indicators = [
        "i cannot provide",
        "no specific",
        "not offer specific",
        "context defines",
        "i don't know",
        "unable to",
        "no information",
        "general purpose",
    ]

    # if it's too short or lacks any pesticide keyword, trigger fallback
    pesticide_keywords = ["spray", "insecticide", "fungicide", "pesticide", "control", "treatment", "management"]

    should_fallback = (
        any(ind in answer.lower() for ind in weak_indicators)
        or not any(word in answer.lower() for word in pesticide_keywords)
        or len(answer) < 80
    )

    if should_fallback:
        try:
            # --- Step 3: Dynamically get Wikipedia info ---
            search_term = query.replace("recommendation", "").replace("control", "").strip()
            wiki_summary = wikipedia.summary(search_term, sentences=6, auto_suggest=True)

            fallback_prompt = (
                "Using the following Wikipedia content, give a concise and factual pesticide "
                "recommendation for controlling pests on apple crops. "
                "Include at least one example pesticide (safe and approved), biological control, "
                "and a short note on Integrated Pest Management (IPM).\n\n"
                f"Wikipedia Context:\n{wiki_summary}\n\n"
                f"User Question: {query}"
            )

            fallback_response = llm.invoke([HumanMessage(content=fallback_prompt)])
            final_answer = f"📚 (Wikipedia Fallback)\n\n{fallback_response.content}"
        except Exception as e:
            final_answer = f"⚠️ Wikipedia Fallback Error: {e}"
    else:
        final_answer = answer

    return {"messages": [AIMessage(content=final_answer)], "context": context_str}



# --- Build Graph ---
@st.cache_resource
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("rag", rag_node)
    g.add_edge("rag", END)
    g.set_entry_point("rag")
    return g.compile()

rag_app = build_graph()

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="👋 Hi! Ask me anything about agriculture or related topics.")]

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

if user_input := st.chat_input("Ask your question here..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.spinner("🔍 Thinking..."):
        response = rag_app.invoke({"messages": st.session_state.messages})

    ai_msg = response["messages"][-1]
    st.session_state.messages.append(ai_msg)
    st.chat_message("assistant").write(ai_msg.content)

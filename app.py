import streamlit as st
import uuid
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from tavily import TavilyClient
import wikipedia

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NAKKAN CHAT BOT ",
    page_icon="💬",
    layout="wide"
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# ---------------- THEME ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111111;
}
h1,h2,h3,p,label {
    color: white !important;
}
.stTextInput input {
    background: #1a1a1a !important;
    color: white !important;
    border: 1px solid #333 !important;
    border-radius: 18px !important;
    padding: 14px !important;
}
div.stButton > button {
    background: #1a1a1a;
    color: white;
    border: 1px solid #333;
    border-radius: 12px;
}
.welcome {
    text-align: center;
    margin-top: 18%;
}
.welcome h1 {
    font-size: 42px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AI ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ChatEasy.
        Use provided live search results whenever available.
        Remember previous conversation in same chat.
        Give accurate and current answers."""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- SESSION ----------------
if "chats" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.chats = {
        cid: {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory()
        }
    }
    st.session_state.current = cid

def get_history(session_id):
    return st.session_state.chats[session_id]["history"]

chat_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ---------------- SEARCH ----------------
def live_search(query):
    try:
        result = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )

        snippets = []
        for item in result["results"]:
            snippets.append(item["content"])

        return "\n".join(snippets)

    except:
        try:
            return wikipedia.summary(query, sentences=3)
        except:
            return ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("ChatEasy")

    if st.button("+ New Chat", use_container_width=True):
        cid = str(uuid.uuid4())
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory()
        }
        st.session_state.current = cid
        st.rerun()

    st.markdown("---")

    delete_id = None

    for cid, chat in st.session_state.chats.items():
        c1, c2 = st.columns([5,1])

        with c1:
            if st.button(chat["title"], key=f"chat_{cid}", use_container_width=True):
                st.session_state.current = cid
                st.rerun()

        with c2:
            if st.button("🗑", key=f"del_{cid}"):
                delete_id = cid

    if delete_id:
        del st.session_state.chats[delete_id]

        if not st.session_state.chats:
            cid = str(uuid.uuid4())
            st.session_state.chats[cid] = {
                "title": "New Chat",
                "messages": [],
                "history": InMemoryChatMessageHistory()
            }

        st.session_state.current = next(iter(st.session_state.chats))
        st.rerun()

# ---------------- MAIN ----------------
current_chat = st.session_state.chats[st.session_state.current]

if current_chat["messages"]:
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything...")

else:
    st.markdown("""
    <div class='welcome'>
        <h1>Hii, what can I help u with Today?</h1>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_input("", placeholder="Ask anything...")

# ---------------- RESPONSE ----------------
if user_input:
    current_chat["messages"].append({
        "role": "user",
        "content": user_input
    })

    if current_chat["title"] == "New Chat":
        current_chat["title"] = user_input[:25]

    live_info = live_search(user_input)

    query = f"""
User Question:
{user_input}

Live Information:
{live_info}

Answer accurately using latest available information.
"""

    with st.spinner("Thinking..."):
        res = chat_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": st.session_state.current}}
        )

    current_chat["messages"].append({
        "role": "assistant",
        "content": res.content
    })

    st.rerun()

# ---------------- CLEAR ----------------
if st.button("Clear Current Conversation"):
    current_chat["messages"] = []
    current_chat["history"] = InMemoryChatMessageHistory()
    st.rerun()

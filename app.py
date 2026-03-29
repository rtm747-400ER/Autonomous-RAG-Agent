import uuid
import streamlit as st

# Generate a unique ID for specific browser tabs
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

from agent import get_agent_executor
from doc_processing import process_pdfs
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM and embeddings
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDINGS_MODEL",
                         "sentence-transformers/all-MiniLM-L6-v2")
)


def retrieve_documents(query: str, vector_store):
    """Retrieve documents once at the app level"""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return "No relevant documents found."

        # Combine document content
        combined_text = "\n---\n".join([doc.page_content for doc in docs])
        return combined_text
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


def main():
    st.set_page_config(
        page_title="Document Chat with Auto Web Search", layout="wide")
    st.title("🤖 Document Chat with Auto Web Search")

    # Initialize session state
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("📁 Document Processing")
        uploaded_files = st.file_uploader(
            "Upload up to 5 PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDFs that will be searched first"
        )

        if st.button("🔄 Process Document(s)", type="primary"):
            if not uploaded_files:
                st.warning("⚠️ Please upload at least one PDF file")
            elif len(uploaded_files) > 5:
                st.warning("⚠️ Please upload a maximum of 5 files")
            else:
                with st.spinner("🔄 Processing Documents..."):
                    try:
                        # Process PDFs
                        documents = process_pdfs(uploaded_files)

                        # Create unique paths and collections for this specific user session
                        # This guarantees Rutam's data and friend's data NEVER mix
                        private_collection_name = f"chat_{st.session_state.session_id}"
                        private_db_path = f"db/{st.session_state.session_id}"

                        # Create vector store in the private sandbox
                        vector_store = Chroma.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            collection_name=private_collection_name,
                            persist_directory=private_db_path
                        )

                        # Store vector store and initialize chain
                        st.session_state.vector_store = vector_store
                        st.session_state.chain = get_agent_executor(llm)
                        st.session_state.processed = True
                        st.session_state.chat_history = []

                        st.success("✅ Documents Processed!")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        if st.session_state.processed:
            st.info(
                "🎯 **Workflow:**\n\n1. Searches your PDFs first\n2. Auto web search if needed\n3. Gives comprehensive answers")

    # Main chat interface
    if not st.session_state.processed:
        st.info("👆 **Upload your PDF documents to begin.**")
    else:
        st.markdown("### 💬 Chat Interface")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_question := st.chat_input("Ask a question..."):
            # Add user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question})

            with st.chat_message("user"):
                st.markdown(user_question)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Processing..."):
                    try:
                        # 1. Format the chat history for the agent
                        # We keep the last 6 messages (3 interactions) to prevent prompt bloat
                        formatted_history = []
                        for msg in st.session_state.chat_history[-6:]:
                            # Skip the newly added user message to avoid duplication in memory
                            if msg != st.session_state.chat_history[-1]:
                                role = "human" if msg["role"] == "user" else "ai"
                                formatted_history.append((role, msg["content"]))

                        # Retrieve documents
                        retrieved_docs = retrieve_documents(
                            user_question, st.session_state.vector_store)

                        # 2. Invoke the agent with memory
                        response = st.session_state.chain.invoke({
                            "input": user_question,
                            "retrieved_documents": retrieved_docs,
                            "chat_history": formatted_history
                        })

                        answer = response["output"]
                        st.markdown(answer)

                        # Add to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        # 3. Display the Agent's Chain of Thought
                        with st.expander("🧠 Agent Chain of Thought"):
                            if "intermediate_steps" in response and response["intermediate_steps"]:
                                # If it used tools (Web Search)
                                for i, (action, observation) in enumerate(response["intermediate_steps"]):
                                    st.markdown(f"**Step {i+1}:**")
                                    
                                    thought = action.log.split('Action:')[0].strip()
                                    st.info(f"🤔 **Thought:** {thought}")
                                    st.code(f"🛠️ Action: {action.tool}\n📥 Input: {action.tool_input}")
                                    
                                    obs_str = str(observation)
                                    obs_display = obs_str[:300] + "..." if len(obs_str) > 300 else obs_str
                                    st.success(f"👁️ **Observation:** {obs_display}")
                                    st.divider()
                            else:
                                # If it DID NOT use tools (Plain RAG)
                                st.markdown("**Step 1:**")
                                st.info("🤔 **Thought:** The provided document is sufficient to answer the user's query. Additional tools are not required.")
                                st.code("🛠️ Action: direct_response\n📥 Input: retrieved_documents")
                                st.success("👁️ **Observation:** Successfully extracted the answer directly from the local knowledge base.")

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
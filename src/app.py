import hashlib

import streamlit as st
import os
import tempfile
import time
import traceback
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, END, StateGraph
from state import AgentState
from decisions import route_decision, grade_documents, should_retry_retrieval
from agents import (
    router_agent,
    retrieve_agent,
    reformulate_agent,
    web_search_agent,
    synthesize_agent,
    generate_agent,
    safety_agent,
    ask_clarification,
)
from langchain_core.messages import HumanMessage, AIMessage
from utils import calculate_knowledge_hash


# Initialize components with configurable parameters
def initialize_system(
    urls: List[str],
    uploaded_files: List[Any],
    chunk_size: int = 250,
    k: int = 3,
    temperature: float = 0.0,
):
    # Load and process documents
    docs = []

    # Load from URLs
    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            st.error(f"Failed to load {url}: {str(e)}")

    # Load from uploaded files
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            if os.path.getsize(temp_file_path) == 0:
                st.error(
                    f"Uploaded file {uploaded_file.name} is empty and was skipped."
                )
                os.unlink(temp_file_path)
                continue

            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".txt":
                loader = TextLoader(temp_file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                os.unlink(temp_file_path)
                continue

            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.name
            docs.extend(file_docs)
            os.unlink(temp_file_path)  # Clean up temp file
        except Exception as e:
            st.error(f"Failed to load uploaded file {uploaded_file.name}: {str(e)}")

    if not docs:
        st.warning("No documents loaded. Using default knowledge sources.")
        default_urls = [
            "https://medium.com/@sridevi.gogusetty/rag-vs-graph-rag-llama-3-1-8f2717c554e6",
            "https://medium.com/@sridevi.gogusetty/retrieval-augmented-generation-rag-gemini-pro-pinecone-1a0a1bfc0534",
            "https://medium.com/@sridevi.gogusetty/introduction-to-ollama-run-llm-locally-data-privacy-f7e4e58b37a0",
            "https://ollama.com/library",
            "https://ai.google.dev/docs/gemini_api_overview",
        ]
        for url in default_urls:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception as e:
                st.error(f"Failed to load default URL {url}: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)

    # Create vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    # The raw retriever instance
    retriever_instance = vectorstore.as_retriever(search_kwargs={"k": k})

    # The retriever_tool that when invoked, returns a *string* of concatenated content
    retriever_tool_for_display = create_retriever_tool(
        retriever_instance,
        "retrieve_knowledge",
        "Search and return information from the provided knowledge sources.",
    )

    web_search_tool = TavilySearch(max_results=5)

    # Build enhanced LangGraph workflow
    workflow = StateGraph(AgentState)

    # Add specialized agent nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("retrieve", retrieve_agent)
    workflow.add_node("reformulate_query", reformulate_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("synthesize", synthesize_agent)
    workflow.add_node("generate", generate_agent)
    workflow.add_node("safety_check", safety_agent)
    workflow.add_node("ask_clarification", ask_clarification)

    # Add edges and conditional routing
    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "relevant": "synthesize",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification",
        },
    )

    workflow.add_conditional_edges(
        "reformulate_query",
        should_retry_retrieval,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )

    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", "generate")
    workflow.add_edge("generate", "safety_check")
    workflow.add_edge("safety_check", END)
    workflow.add_edge("ask_clarification", END)

    # Return the raw retriever_instance for use in retrieve_agent
    return (
        workflow.compile(),
        retriever_instance,
        web_search_tool,
        temperature,
        retriever_tool_for_display,
    )



def main():
    # ========================
    # STREAMLIT APP
    # ========================
    st.set_page_config(page_title="Multi-Agent RAG System", layout="wide")
    st.title("🤖 Advanced Agentic RAG System")
    st.caption("Multi-agent collaboration with self-correction and memory")

    # Recommendation for better results
    st.info(
        "🔎 **Recommendation:** For the most accurate and relevant answers, please explicitly add your own URLs or upload documents as knowledge sources. "
        "Relying only on default sources may yield less tailored results."
    )
    with st.sidebar:
        st.markdown(f"""
        **Current Settings:**  
        - Chunk Size: `{st.session_state.get("chunk_size", 250)}`
        - Retriever K: `{st.session_state.get("retriever_k", 3)}`
        - Temperature: `{st.session_state.get("temperature", 0.0)}`
        """)
        with st.expander("⚙️ Adjust Configuration", expanded=False):
            chunk_size = st.slider(
                "Text Chunk Size", 100, 1000, st.session_state.get("chunk_size", 250), 50
            )
            retriever_k = st.slider(
                "Retriever K Value (Top K Docs)",
                1,
                10,
                st.session_state.get("retriever_k", 3),
            )
            temperature = st.slider(
                "LLM Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.0), 0.1
            )
            st.divider()
        st.subheader("Knowledge Sources")
        st.write("Add URLs (one per line):")
        url_input = st.text_area(
            "Enter URLs", height=150, value="", label_visibility="collapsed"
        )
        uploaded_files = st.file_uploader(
            "Upload text files", type=["txt", "pdf", "docx"], accept_multiple_files=True
        )
        reset_params = st.button("Apply Parameters & Update Knowledge")
        st.divider()
        st.subheader("Agent Roles")
        st.markdown("""
        - **Router**: Determines workflow path based on query type.
        - **Retriever**: Fetches relevant documents from internal knowledge base.
        - **Reformulator**: Improves queries for better internal retrieval.
        - **Web Searcher**: Finds real-time, external information.
        - **Synthesizer**: Combines information from various sources.
        - **Generator**: Creates the final answer.
        - **Safety Checker**: Ensures generated content is safe and appropriate.
        - **Clarification**: Asks user for more details if needed.
        """)

    # Initialize session state
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""
    if "params_applied" not in st.session_state:
        st.session_state.params_applied = False
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "knowledge_hash" not in st.session_state:
        st.session_state.knowledge_hash = ""
    if "retrieved_docs_display" not in st.session_state:
        st.session_state.retrieved_docs_display = []


    # Calculate current knowledge hash
    def calculate_knowledge_hash(urls, files):
        content = "|".join(sorted(urls))
        for file in files:
            content += file.getvalue().decode(errors="ignore")
        return hashlib.md5(content.encode()).hexdigest()


    # Parse URLs from input
    urls = [url.strip() for url in url_input.split("\n") if url.strip()]

    # Check if knowledge has changed
    current_knowledge_hash = calculate_knowledge_hash(urls, uploaded_files)
    knowledge_changed = current_knowledge_hash != st.session_state.knowledge_hash

    # Initialize or update system
    if reset_params or not st.session_state.params_applied or knowledge_changed:
        with st.spinner("Configuring system with new parameters and knowledge sources..."):
            try:
                (
                    st.session_state.graph,
                    st.session_state.retriever_instance,
                    st.session_state.web_search_tool,
                    st.session_state.temperature,
                    st.session_state.retriever_tool_for_display,
                ) = initialize_system(
                    urls=urls,
                    uploaded_files=uploaded_files,
                    chunk_size=chunk_size,
                    k=retriever_k,
                    temperature=temperature,
                )
                st.session_state.params_applied = True
                st.session_state.knowledge_hash = current_knowledge_hash
                st.success("System configured with new knowledge!")
            except Exception as e:
                st.error(f"Configuration failed: {str(e)}")
                st.stop()

    # Chat interface
    with st.container():
        st.subheader("Chat")
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.write(msg.content)

        if prompt := st.chat_input("Ask about your knowledge sources..."):
            # Add user message to history
            user_msg = HumanMessage(content=prompt)
            st.session_state.chat_history.append(user_msg)

            with st.chat_message("user"):
                st.write(prompt)

            # Prepare agent state
            agent_state = AgentState(
                messages=[user_msg],
                chat_history=st.session_state.chat_history,
                current_query=prompt,
                reformulation_count=0,
                retrieved_docs=[],
            )

            # Execute graph
            with st.spinner("Executing multi-agent workflow..."):
                st.session_state.logs = [f"New query: {prompt}"]
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    step_count = 0
                    max_steps = 15
                    current_state = agent_state

                    # Stream outputs from the graph
                    for output in st.session_state.graph.stream(agent_state):
                        node_name = list(output.keys())[0]
                        node_state = output[node_name]

                        status_text.info(
                            f"Executing: **{node_name.replace('_', ' ').title()}**"
                        )
                        st.session_state.logs.append(f"Completed node: {node_name}")

                        step_count += 1
                        progress_bar.progress(min(step_count / max_steps, 1.0))
                        time.sleep(0.3)

                        # Store the latest state updates
                        current_state_updates = node_state

                        if step_count >= max_steps:
                            st.session_state.logs.append(
                                "⚠️ Safety break: Exceeded maximum steps. Ending workflow."
                            )
                            break

                    progress_bar.empty()
                    status_text.success("✅ Workflow completed!")

                    # Process final output
                    if (
                        current_state_updates
                        and "generated_answer" in current_state_updates
                    ):
                        final_answer = current_state_updates["generated_answer"]
                        ai_msg = AIMessage(content=final_answer)
                        st.session_state.chat_history.append(ai_msg)
                        st.session_state.final_answer = final_answer

                        with st.chat_message("assistant"):
                            st.write(final_answer)
                    else:
                        fallback_message = "I couldn't generate a complete response for your query. Please try rephrasing."
                        ai_msg = AIMessage(content=fallback_message)
                        st.session_state.chat_history.append(ai_msg)
                        st.session_state.final_answer = fallback_message

                        with st.chat_message("assistant"):
                            st.write(fallback_message)

                except Exception as e:
                    import traceback

                    error_trace = traceback.format_exc()
                    status_text.error(f"❌ Execution failed: {str(e)}")
                    st.session_state.logs.append(f"ERROR TRACEBACK:\n{error_trace}")
                    error_msg = AIMessage(
                        content="Sorry, I encountered an error processing your request. Please check the logs."
                    )
                    st.session_state.chat_history.append(error_msg)

                    with st.chat_message("assistant"):
                        st.write(
                            "Sorry, I encountered an error processing your request. Please check the logs."
                        )

    # Display retrieved documents
    if st.session_state.retrieved_docs_display:
        with st.expander("📄 Retrieved Documents", expanded=False):
            for i, doc in enumerate(st.session_state.retrieved_docs_display, 1):
                st.subheader(f"Document {i}")
                source = doc.get("metadata", {}).get("source", "Unknown")
                st.caption(f"Source: {source}")
                st.write(doc.get("content", ""))
                st.divider()
    else:
        st.info("📄 No documents retrieved yet. Ask a question to see relevant documents.")

    # Display results
    with st.expander("Execution Details", expanded=False):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Workflow Log")
            log_container = st.container(height=300)
            for log in st.session_state.logs:
                log_container.code(log, language="log")

            st.subheader("Active Knowledge Sources")
            if urls or uploaded_files:
                st.markdown("**URLs:**")
                for url in urls:
                    st.markdown(f"- [{url}]({url})")

                if uploaded_files:
                    st.markdown("**Uploaded Files:**")
                    for file in uploaded_files:
                        st.markdown(f"- {file.name}")
            else:
                st.markdown("Using default knowledge sources")

        with col2:
            st.subheader("Workflow Diagram")
            st.graphviz_chart("""
                digraph {
                    node [shape=box, style=rounded]
                    start -> router
                    router -> retrieve [label="retrieve"]
                    router -> reformulate_query [label="reformulate"]
                    router -> web_search [label="web_search"]
                    router -> ask_clarification [label="clarify"]
                    router -> generate [label="generate (simple query)"]
                    
                    retrieve -> grade_documents [label="docs retrieved"]
                    
                    grade_documents -> synthesize [label="relevant"]
                    grade_documents -> reformulate_query [label="reformulate"]
                    grade_documents -> web_search [label="irrelevant / no docs"]
                    grade_documents -> ask_clarification [label="clarify"]
                    
                    reformulate_query -> retrieve [label="retry retrieval (max 2)"]
                    reformulate_query -> web_search [label="fallback to web search"]
                    
                    web_search -> synthesize [label="results found"]
                    
                    synthesize -> generate [label="summary created"]
                    generate -> safety_check [label="answer created"]
                    safety_check -> end [label="content safe"]
                    
                    ask_clarification -> end [label="user asked to clarify"]
                }
            """)

            st.subheader("Agent Configuration")
            st.markdown(f"""
            - **Chunk Size**: `{chunk_size}`
            - **Retriever K (Top K Docs)**: `{retriever_k}`
            - **LLM Temperature**: `{temperature}`
            - **Main LLM Model**: `openai/gpt-oss-120b` (Groq)
            - **Grading LLM Model**: `openai/gpt-oss-120b` (Groq)
            """)

    # Add reset button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.logs = []
        st.session_state.final_answer = ""
        st.session_state.retrieved_docs_display = []
        st.session_state.get("knowledge_hash", None)  # Clear knowledge hash
        st.session_state.params_applied = False
        st.rerun()

    st.divider()
    st.caption(
        "Advanced Agentic RAG System | Multi-agent collaboration with self-correction"
    )

if __name__ == "__main__":
    main()

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from state import AgentState
import re


# ========================
# SPECIALIZED AGENT FUNCTIONS
# ========================


def router_agent(state: AgentState) -> dict:
    """Decide initial workflow path based on query and history"""
    st.session_state.logs.append("---ROUTER AGENT---")

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""As the Router Agent, analyze the user's question and conversation history to determine the best next step.
        
        Conversation History:
        {history}
        
        Current Question: {question}
        
        Choose one of these actions:
        - "retrieve": Use this for questions about knowledge base content, definitions, explanations, or factual information that could be in the documents. This is the default for most questions.
        - "reformulate": Use only if the query is complex and needs refinement for better retrieval
        - "web_search": Use only if the question requires current/realtime information or broader knowledge beyond internal documents
        - "clarify": Use only if the question is too vague or needs more details
        - "generate": Use ONLY for very simple conversational responses like "hello", "thank you", "how are you", or basic greetings
        
        IMPORTANT: Default to "retrieve" for all factual or knowledge-based questions. Only use "generate" for basic social interactions.
        
        Only respond with the action word.""",
        input_variables=["question", "history"],
    )

    history_str = "\n".join([f"{m.type}: {m.content}" for m in state.chat_history[-5:]])
    response = model.invoke(
        prompt.format(question=state.current_query, history=history_str)
    )
    decision = response.content.strip().lower()

    st.session_state.logs.append(f"Routing decision: {decision}")
    return {"next_step": decision}


def retrieve_agent(state: AgentState) -> dict:
    """Retrieve relevant documents from vector store"""
    st.session_state.logs.append("---RETRIEVAL AGENT---")
    query = state.current_query

    try:
        # Use invoke() instead of get_relevant_documents() for VectorStoreRetriever
        docs_list_objects = st.session_state.retriever_instance.invoke(query)

        # Convert Document objects to a list of dicts for Pydantic AgentState compatibility
        retrieved_content_with_meta = []
        for doc in docs_list_objects:
            retrieved_content_with_meta.append(
                {"content": doc.page_content, "metadata": doc.metadata}
            )

        st.session_state.logs.append(
            f"Retrieved {len(retrieved_content_with_meta)} documents"
        )
        # Store retrieved documents for display in UI
        st.session_state.retrieved_docs_display = retrieved_content_with_meta
        return {"retrieved_docs": retrieved_content_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"Retrieval error: {str(e)}")
        return {"retrieved_docs": []}  # Ensure it's an empty list on error


def reformulate_agent(state: AgentState) -> dict:
    """Reformulate query for better retrieval"""
    st.session_state.logs.append("---QUERY REFORMULATION AGENT---")

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""Reformulate the user's query to improve document retrieval. Consider:
        - Synonyms and related terms
        - Breaking down complex questions
        - Context from conversation history
        
        Original Query: {query}
        Conversation History: {history}
        Previous Retrieval Results (if any, truncated): {previous_results}
        
        Respond ONLY with the reformulated query.""",
        input_variables=["query", "history", "previous_results"],
    )

    history_str = "\n".join([f"{m.type}: {m.content}" for m in state.chat_history[-3:]])
    # Extract content from retrieved_docs for previous_results display
    previous_results_content = (
        "\n\n".join([doc["content"] for doc in state.retrieved_docs])
        if state.retrieved_docs
        else ""
    )

    response = model.invoke(
        prompt.format(
            query=state.current_query,
            history=history_str,
            previous_results=previous_results_content[:500]
            if previous_results_content
            else "None",  # Truncate for prompt
        )
    )

    new_query = response.content.strip()
    st.session_state.logs.append(f"Reformulated query: {new_query}")
    return {
        "current_query": new_query,
        "reformulation_count": state.reformulation_count + 1,
    }


def web_search_agent(state: AgentState) -> dict:
    """Perform web search using Tavily"""
    st.session_state.logs.append("---WEB SEARCH AGENT---")
    query = state.current_query

    try:
        results = st.session_state.web_search_tool.invoke({"query": query})

        # Handle different result formats from TavilySearch
        web_results_with_meta = []

        # If results is a string, create a single document from it
        if isinstance(results, str):
            web_results_with_meta.append(
                {"content": results, "metadata": {"source": "web_search"}}
            )
        # If results is a list, process each item
        elif isinstance(results, list):
            for d in results:
                if isinstance(d, dict):
                    # If dict has 'content' or 'result', use that
                    content = d.get("content") or d.get("result") or str(d)
                    web_results_with_meta.append(
                        {
                            "content": content,
                            "metadata": d.get("metadata", {"source": "web_search"}),
                        }
                    )
                elif isinstance(d, str):
                    # If it's a string, create a simple document
                    web_results_with_meta.append(
                        {"content": d, "metadata": {"source": "web_search"}}
                    )

        st.session_state.logs.append(f"Found {len(web_results_with_meta)} web results")
        # Store web search results for display in UI
        st.session_state.retrieved_docs_display = web_results_with_meta
        return {"retrieved_docs": web_results_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"Web search error: {str(e)}")
        return {"retrieved_docs": []}  # Ensure it's an empty list on error


def synthesize_agent(state: AgentState) -> dict:
    """Synthesize information from multiple sources"""
    st.session_state.logs.append("---SYNTHESIS AGENT---")

    if not state.retrieved_docs:
        st.session_state.logs.append("No documents for synthesis.")
        return {"retrieved_docs": []}  # Return empty list if no docs

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""Synthesize key information from these knowledge sources:
        
        {sources}
        
        Focus on answering: {question}
        
        Extract and combine the most relevant facts. Omit irrelevant details.
        Respond with a concise knowledge summary. Include source attribution where possible.
        """,
        input_variables=["sources", "question"],
    )

    # Extract content from the list of dicts for the sources prompt
    sources_text = []
    for i, doc in enumerate(state.retrieved_docs):
        source_info = doc["metadata"].get("source", f"Document {i + 1}")
        sources_text.append(f"Source {i + 1} ({source_info}):\n{doc['content']}")
    sources_content_for_llm = "\n\n".join(sources_text)

    response = model.invoke(
        prompt.format(sources=sources_content_for_llm, question=state.current_query)
    )

    synthesized = response.content
    st.session_state.logs.append(f"Synthesized {len(synthesized.split())} word summary")
    # Store synthesized content as a single document-like dict for generation.
    return {
        "retrieved_docs": [
            {"content": synthesized, "metadata": {"source": "synthesized_knowledge"}}
        ]
    }


def generate_agent(state: AgentState) -> dict:
    """Generate final answer from context"""
    st.session_state.logs.append("---GENERATION AGENT---")

    if not state.retrieved_docs:
        st.session_state.logs.append("No context available for generation.")
        return {
            "generated_answer": "I don't have enough information to answer that question."
        }

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )
    rag_chain = prompt | model | StrOutputParser()

    # Extract content from the list of dicts for the context
    context_content = "\n\n".join([doc["content"] for doc in state.retrieved_docs])

    response = rag_chain.invoke(
        {"context": context_content, "question": state.current_query}
    )

    st.session_state.logs.append("Response generated")
    return {"generated_answer": response}


def fact_check_agent(state: AgentState) -> dict:
    """Fact-check generated answer using web search"""
    st.session_state.logs.append("---FACT-CHECK AGENT---")

    # Ensure generated_answer is not None before proceeding
    if not state.generated_answer:
        st.session_state.logs.append("No generated answer to fact-check.")
        return {"generated_answer": state.generated_answer or ""}

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    claims_prompt = """Identify factual claims in this text. List them as bullet points:
    
    {text}
    
    Respond ONLY with the bullet points of claims. Each claim should start with a hyphen '- '."""

    claims_response = model.invoke(
        claims_prompt.format(text=state.generated_answer)
    ).content
    # Filter out empty lines or lines not starting with a bullet point
    claims_list = [
        c.strip()
        for c in claims_response.split("\n")
        if c.strip() and c.lstrip().startswith("-")
    ]
    st.session_state.logs.append(f"Identified {len(claims_list)} claims to verify")

    # Verify each claim with web search
    verified_claims = []
    # Limit to top 3 for efficiency, and ensure there are claims to check
    for claim in claims_list[:3]:
        try:
            # Remove leading hyphen if present before searching
            search_query = claim[2:] if claim.startswith("- ") else claim
            results = st.session_state.web_search_tool.invoke(
                {"query": f"Verify: {search_query}"}
            )

            # Handle different result formats from TavilySearch
            sources = []
            if isinstance(results, str):
                # If results is a string, use it directly
                sources = [results]
            elif isinstance(results, list):
                # If results is a list, try to extract content from each item
                for d in results:
                    if isinstance(d, dict) and "content" in d:
                        sources.append(d["content"])
                    elif isinstance(d, str):
                        sources.append(d)
                    elif isinstance(d, dict) and "result" in d:
                        sources.append(d["result"])
                    elif isinstance(d, dict) and "answer" in d:
                        sources.append(d["answer"])

            sources = sources[:2]  # Limit to top 2 sources

            verification_prompt = f"""Based on these sources, is this claim true?
            Claim: {claim}
            
            Sources:
            {sources}
            
            Respond with "TRUE" or "FALSE" and a brief explanation."""

            verdict = model.invoke(verification_prompt).content
            verified_claims.append(f"{claim} → {verdict}")
        except Exception as e:
            st.session_state.logs.append(f"Fact-check for '{claim}' failed: {str(e)}")
            verified_claims.append(f"{claim} → Verification failed (Error: {str(e)})")

    # Update answer with verification notes
    if verified_claims:
        verified_text = "\n".join(verified_claims)
        updated_answer = (
            f"{state.generated_answer}\n\n**Fact Check Results:**\n{verified_text}"
        )
    else:
        updated_answer = (
            state.generated_answer
            + "\n\n**Fact Check:** No specific claims identified or verified."
        )

    st.session_state.logs.append(f"Verified {len(verified_claims)} claims")
    return {"generated_answer": updated_answer}


def safety_agent(state: AgentState) -> dict:
    """Check for harmful/inappropriate content"""
    st.session_state.logs.append("---SAFETY AGENT---")

    if not state.generated_answer:
        st.session_state.logs.append("No generated answer to safety check.")
        return {"generated_answer": state.generated_answer or ""}

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""Analyze this text for harmful, biased, or inappropriate content:
        
        {text}
        
        Respond in this exact format:
        Safety Rating: [SAFE/CONCERN/UNSAFE]
        Issues: [List any issues found, or 'None' if safe]
        Revised Text: [If issues found and revisable, provide a revised version. Otherwise, state 'Not revisable'.]
        
        Example for unsafe:
        Safety Rating: UNSAFE
        Issues: Contains hate speech
        Revised Text: Not revisable
        
        Example for safe:
        Safety Rating: SAFE
        Issues: None
        Revised Text: N/A""",
        input_variables=["text"],
    )

    response_content = model.invoke(prompt.format(text=state.generated_answer)).content

    safety_rating_match = re.search(
        r"Safety Rating:\s*\[?(SAFE|CONCERN|UNSAFE)\]?", response_content, re.IGNORECASE
    )
    safety_rating = (
        safety_rating_match.group(1).upper() if safety_rating_match else "UNKNOWN"
    )

    # Use re.DOTALL to match across multiple lines for Revised Text
    revised_text_match = re.search(
        r"Revised Text:\s*(.*?)(?=\n[A-Z]|$)",
        response_content,
        re.DOTALL | re.IGNORECASE,
    )
    revised_text = revised_text_match.group(1).strip() if revised_text_match else "N/A"

    final_answer_after_safety = state.generated_answer  # Default to original

    if safety_rating == "SAFE":
        final_answer_after_safety = state.generated_answer
    elif "Not revisable" in revised_text:
        final_answer_after_safety = (
            "I cannot answer that question due to safety concerns."
        )
    elif revised_text and revised_text != "N/A":
        final_answer_after_safety = revised_text

    st.session_state.logs.append(f"Safety rating: {safety_rating}")
    return {"generated_answer": final_answer_after_safety}


def ask_clarification(state: AgentState) -> dict:
    """Ask user for clarification"""
    st.session_state.logs.append("---ASKING FOR CLARIFICATION---")

    clarification = "I need a bit more clarity to answer your question effectively. Could you please rephrase or provide more details?"
    return {"generated_answer": clarification}

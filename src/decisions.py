import streamlit as st
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from state import AgentState
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


# ========================
# DECISION FUNCTIONS
# ========================


def route_decision(state: AgentState) -> str:
    """Determine next step after routing"""
    # This function simply returns the decision made by the router_agent
    return state.next_step


def grade_documents(
    state: AgentState,
) -> Literal["relevant", "reformulate", "web_search", "clarify"]:
    """Evaluate document relevance and suggest next action"""
    st.session_state.logs.append("---GRADE DOCUMENTS---")

    # If no documents were retrieved at all, they are irrelevant
    if not state.retrieved_docs:
        st.session_state.logs.append(
            "No documents retrieved, defaulting action to 'web_search'."
        )
        return "web_search"

    model = ChatGroq(
        temperature=st.session_state.temperature,
        model_name="openai/gpt-oss-20b",
    )

    prompt = PromptTemplate(
        template="""Evaluate retrieved documents for question relevance:
        
        Question: {question}
        Documents: {context}
        
        Score options:
        - relevant: Directly answers question
        - partial: Partially relevant but incomplete, may need more info or different sources
        - irrelevant: Not relevant at all
        
        Action options (based on score and need for more info):
        - synthesize: Use documents as-is to synthesize an answer
        - reformulate: Improve query and retry internal retrieval (if partial/irrelevant but internal knowledge might exist)
        - web_search: Use web search instead (if irrelevant or query is external-facing)
        - clarify: Ask user for clarification (if question is ambiguous)
        
        Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
        {{"score": "relevant", "action": "synthesize"}}
        
        Valid values for score: "relevant", "partial", "irrelevant"
        Valid values for action: "synthesize", "reformulate", "web_search", "clarify"
        """,
        input_variables=["question", "context"],
    )

    # Extract content from the list of dicts for the context
    context_content = "\n\n".join([doc["content"] for doc in state.retrieved_docs])

    try:
        response = model.invoke(
            prompt.format(
                question=state.current_query,
                context=context_content[:4000],
            )
        ).content

        st.session_state.logs.append(f"Grading response: {response}")

        # Try to parse JSON from response
        import json
        import re

        # Extract JSON from response (handles cases where model wraps in markdown)
        json_match = re.search(
            r"\{[^{}]*\"score\"[^{}]*\"action\"[^{}]*\}", response, re.DOTALL
        )
        if json_match:
            try:
                result = json.loads(json_match.group())
                score = result.get("score", "").lower()
                action = result.get("action", "").lower()

                st.session_state.logs.append(f"Relevance: {score} → Action: {action}")

                # Map 'synthesize' to 'relevant' since that's what the router expects
                if action == "synthesize":
                    return "relevant"
                if action in ["relevant", "reformulate", "web_search", "clarify"]:
                    return action

                # Fallback based on score if action is invalid
                if score == "relevant":
                    return "relevant"
                elif score == "partial":
                    return "reformulate"
                else:
                    return "web_search"
            except json.JSONDecodeError:
                st.session_state.logs.append("JSON parse failed, using heuristic")
        else:
            st.session_state.logs.append("No JSON found in response, using heuristic")

        # Heuristic fallback if JSON parsing fails
        response_lower = response.lower()
        if "relevant" in response_lower and (
            ("synthes" in response_lower) or ("answer" in response_lower)
        ):
            return "relevant"
        elif "reformulate" in response_lower or "partial" in response_lower:
            return "reformulate"
        elif "clarify" in response_lower or "ambiguous" in response_lower:
            return "clarify"
        else:
            return "web_search"

    except Exception as e:
        st.session_state.logs.append(
            f"Grading error: {str(e)}. Defaulting to 'web_search'."
        )
        return "web_search"


def should_retry_retrieval(state: AgentState) -> Literal["retrieve", "web_search"]:
    """Decide whether to retry internal retrieval after reformulation or go to web search."""
    # This limits the number of query reformulations before falling back to web search
    if state.reformulation_count < 2:  # Allow up to 2 reformulations
        st.session_state.logs.append(
            f"Reformulation count: {state.reformulation_count}. Retrying retrieval."
        )
        return "retrieve"
    st.session_state.logs.append(
        f"Reformulation limit reached ({state.reformulation_count}). Falling back to web search."
    )
    return "web_search"

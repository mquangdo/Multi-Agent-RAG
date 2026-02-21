from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict


# Extended state class with chat history and agent-specific states
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)
    reformulation_count: int = 0
    current_query: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    generated_answer: Optional[str] = None
    next_step: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

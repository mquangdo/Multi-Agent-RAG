__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, Literal, Annotated, Optional, Tuple, Dict, Any
from langchain_core.prompts import PromptTemplate
import json
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import time
import re
import tempfile
import hashlib
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

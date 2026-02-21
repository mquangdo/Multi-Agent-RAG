# Agentic RAG System

Hệ thống RAG thông minh sử dụng multi-agent architecture với LangGraph và Streamlit.

## Tính năng

- Multi-agent routing: Tự động chọn cách tìm kiếm phù hợp
- Query reformulation: Tự động cải thiện câu hỏi để kết quả tốt hơn
- Web search fallback: Tìm kiếm từ web khi tài liệu không đủ
- Document grading: Đánh giá độ liên quan của tài liệu
- Safety check: Kiểm tra nội dung an toàn
- Hỗ trợ nhiều nguồn: URL, TXT, PDF, DOCX

## Cài đặt

1. Tạo file `.env`:
```
GROQ_API_KEY=key_cua_ban
TAVILY_API_KEY=key_cua_ban
LANGSMITH_API_KEY=key_cua_ban
```

2. Cài dependencies:
```bash
pip install streamlit langchain langgraph langchain-groq langchain-tavily langchain-huggingface langchain-community chromadb sentence-transformers
```

3. Chạy:
```bash
streamlit run app.py
```

## Cách dùng

1. Thêm URL hoặc upload file vào sidebar
2. Nhấn "Apply Parameters & Update Knowledge"
3. Đặt câu hỏi và nhận câu trả lời

## Cấu trúc dự án

- `app.py`: Main app (cấu trúc dạng module)
- `main.py`: Main app (single file)
- `agents.py`: Các agent functions
- `decisions.py`: Logic routing
- `state.py`: State management
- `config.py`: Configuration
- `utils.py`: Utility functions

## Công nghệ sử dụng

- LangGraph: Multi-agent orchestration
- Streamlit: UI
- Groq: LLM
- HuggingFace: Embeddings
- ChromaDB: Vector store
- Tavily: Web search

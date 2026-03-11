"""RAG Q&A pipeline for CatchUp."""

from rag.qa_chain import QAResult, SourceBlock, index_document, query

__all__ = ["index_document", "query", "QAResult", "SourceBlock"]

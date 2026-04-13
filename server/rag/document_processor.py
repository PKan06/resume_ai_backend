# server/rag/document_processor.py
import os
import logging
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        
    def load_raw(self) -> list[Document]:
        """Load and clean docs WITHOUT splitting — for profile extraction."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Resume file not found: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        cleaned_docs = []
        for d in docs:
            text = re.sub(r"\s+", " ", d.page_content).strip()
            if text:
                d.page_content = text
                cleaned_docs.append(d)

        return cleaned_docs

    def load_and_process(self) -> list[Document]:

        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Resume file not found: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        # =========================
        # CLEAN TEXT
        # =========================
        cleaned_docs = []
        for d in docs:
            text = re.sub(r"\s+", " ", d.page_content).strip()
            if text:
                d.page_content = text
                cleaned_docs.append(d)

        # =========================
        # 🔥 MODERN SPLITTING (KEY CHANGE)
        # =========================
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,          # 🔥 smaller → sharper embeddings
            chunk_overlap=80,
            separators=[
                "\n\n",
                "\n",
                ". ",
                ", ",
                " "
            ]
        )

        split_docs = splitter.split_documents(cleaned_docs)

        # =========================
        # 🔥 CONTEXT ENRICHMENT
        # =========================
        enriched_docs = []

        for i, doc in enumerate(split_docs):

            text = doc.page_content.lower()

            # 🔥 semantic tagging
            if "python" in text or "fastapi" in text:
                section = "backend"
            elif "react" in text or "frontend" in text:
                section = "frontend"
            elif "project" in text:
                section = "project"
            elif "education" in text:
                section = "education"
            else:
                section = "general"

            # 🔥 KEY CHANGE → natural language prefix (NOT brackets)
            enriched_text = f"""
                This document contains information about the candidate's {section}.

                Key details:
                {doc.page_content}
                """

            doc.page_content = enriched_text
            doc.metadata["section"] = section
            doc.metadata["chunk_id"] = i

            enriched_docs.append(doc)

        logger.info(f"✅ Created {len(enriched_docs)} optimized chunks")

        return enriched_docs
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

    # =========================
    # 🔥 CLEAN FUNCTION (NEW)
    # =========================
    def _clean_text(self, text: str) -> str:
        # remove hard line breaks
        text = re.sub(r"\n+", " ", text)

        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        # fix spacing before punctuation
        text = re.sub(r"\s+([.,])", r"\1", text)

        return text.strip()

    def load_raw(self) -> list[Document]:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Resume file not found: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        cleaned_docs = []
        for d in docs:
            text = self._clean_text(d.page_content)
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
            text = self._clean_text(d.page_content)
            if text:
                d.page_content = text
                cleaned_docs.append(d)

        # =========================
        # SPLITTING
        # =========================
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )

        split_docs = splitter.split_documents(cleaned_docs)

        # =========================
        # ENRICHMENT (FIXED)
        # =========================
        enriched_docs = []

        for i, doc in enumerate(split_docs):

            # 🔥 CLEAN AGAIN AFTER SPLIT
            clean_chunk = self._clean_text(doc.page_content)

            text = clean_chunk.lower()

            # semantic tagging
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

            # 🔥 FIXED (space added)
            enriched_text = (
                f"This section describes the candidate's {section}. "
                f"{clean_chunk}"
            )

            doc.page_content = enriched_text
            doc.metadata["section"] = section
            doc.metadata["chunk_id"] = i

            enriched_docs.append(doc)

        logger.info(f"✅ Created {len(enriched_docs)} optimized chunks")

        return enriched_docs
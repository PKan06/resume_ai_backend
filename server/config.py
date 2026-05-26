import os
from dotenv import load_dotenv

load_dotenv()


class SettingsError(ValueError):
    """Raised when required application settings are missing or malformed."""


class ConfigParser:
    def get_required(self, name: str) -> str:
        value = os.getenv(name)
        if value is None or not value.strip():
            raise SettingsError(f"Missing required environment variable: {name}")
        return value.strip()

    def get_optional(self, name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value is None:
            return default
        value = value.strip()
        return value or default

    def get_int(self, name: str, default: int | None = None) -> int:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            if default is None:
                raise SettingsError(f"Missing required integer environment variable: {name}")
            return default

        try:
            return int(raw.strip())
        except ValueError as exc:
            raise SettingsError(
                f"Invalid integer for environment variable {name}: {raw!r}"
            ) from exc

    def get_float(self, name: str, default: float | None = None) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            if default is None:
                raise SettingsError(f"Missing required float environment variable: {name}")
            return default

        try:
            return float(raw.strip())
        except ValueError as exc:
            raise SettingsError(
                f"Invalid float for environment variable {name}: {raw!r}"
            ) from exc

    def get_csv(self, name: str, default: list[str] | None = None) -> list[str]:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return list(default or [])

        values = [item.strip() for item in raw.split(",") if item.strip()]
        return values or list(default or [])


class Settings:
    def __init__(self):
        parser = ConfigParser()

        # =========================
        # API KEYS
        # =========================
        self.OPENROUTER_API_KEY = parser.get_optional("OPENROUTER_API_KEY")
        self.OPENROUTER_BASE_URL = parser.get_optional("OPENROUTER_BASE_URL")
        self.GEMINI_API_KEY = parser.get_optional("GEMINI_API_KEY")

        # =========================
        # MODEL CONFIG
        # =========================
        self.MODEL_NAME = parser.get_required("MODEL_NAME")
        self.PDF_PATH = parser.get_required("PDF_PATH")

        # =========================
        # CLOUDFLARE CONFIG
        # =========================
        self.CLOUDFLARE_ACCOUNT_ID = parser.get_required("CLOUDFLARE_ACCOUNT_ID")
        self.CLOUDFLARE_API_TOKEN = parser.get_required("CLOUDFLARE_API_TOKEN")
        self.CLOUDFLARE_EMBEDDING_MODEL = parser.get_optional(
            "CLOUDFLARE_EMBEDDING_MODEL",
            default="@cf/baai/bge-small-en-v1.5",
        )

        # =========================
        # PINECONE CONFIG
        # =========================
        self.PINECONE_API_KEY = parser.get_required("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = parser.get_optional("PINECONE_INDEX_NAME", default="resume-ai")
        self.PINECONE_FORCE_REINDEX = parser.get_optional("PINECONE_FORCE_REINDEX", default="false")

        # =========================
        # RETRIEVAL CONFIG
        # =========================
        self.RETRIEVAL_TOP_K = parser.get_int("RETRIEVAL_TOP_K", default=3)
        self.RAG_SCORE_THRESHOLD = parser.get_float(
            "RAG_SCORE_THRESHOLD",
            default=0.35,
        )

        # =========================
        # ASSISTANT CONFIG
        # =========================
        self.ASSISTANT_NAME = parser.get_optional(
            "ASSISTANT_NAME",
            default="Alex AI",
        )
        self.ALLOWED_ORIGINS = parser.get_csv(
            "ALLOWED_ORIGINS",
            default=["http://localhost:3000"],
        )

        # =========================
        # LANGCHAIN (OPTIONAL)
        # =========================
        self.LANGCHAIN_TRACING_V2 = parser.get_optional(
            "LANGCHAIN_TRACING_V2",
            default="true",
        )
        self.LANGCHAIN_API_KEY = parser.get_optional("LANGCHAIN_API_KEY")
        self.LANGCHAIN_PROJECT = parser.get_optional(
            "LANGCHAIN_PROJECT",
            default="resume-assistant",
        )
        self.LANGSMITH_ENDPOINT = parser.get_optional("LANGSMITH_ENDPOINT")

        # =========================
        # REDIS CONFIG (OPTIONAL)
        # =========================
        self.REDIS_HOST = parser.get_optional("REDIS_HOST")
        self.REDIS_PORT = parser.get_int("REDIS_PORT", default=6379)
        self.REDIS_PASSWORD = parser.get_optional("REDIS_PASSWORD")
        self.REDIS_ENABLED = bool(self.REDIS_HOST)

        # The current LLM adapter uses Gemini directly, so fail fast when it
        # is not configured instead of letting background assistant load fail.
        self._require_llm_credentials()

    def _require_llm_credentials(self) -> None:
        if self.GEMINI_API_KEY:
            return

        raise SettingsError(
            "Missing required environment variable: GEMINI_API_KEY. "
            "The current backend LLM adapter uses Gemini directly."
        )

# =========================
# INSTANCE
# =========================
settings = Settings()

# =========================
# LANGSMITH GLOBALS
# =========================
LANGCHAIN_TRACING_V2 = settings.LANGCHAIN_TRACING_V2
LANGCHAIN_API_KEY = settings.LANGCHAIN_API_KEY
LANGCHAIN_PROJECT = settings.LANGCHAIN_PROJECT
LANGSMITH_ENDPOINT = settings.LANGSMITH_ENDPOINT

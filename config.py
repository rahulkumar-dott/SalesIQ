"""Load settings from .env into os.environ (no extra dependency).

Real environment variables win over .env, so CI or `setx` values are never overridden.
Import this module early:  import config  # noqa: F401
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()


def llm_backend() -> str:
    """'azure_openai' (OpenAI model on Azure), 'foundry' (Claude on Azure),
    'anthropic' (direct API) or 'none'. LLM_PROVIDER in .env forces one."""
    forced = os.environ.get("LLM_PROVIDER")
    if forced:
        return forced
    key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
    res = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_RESOURCE")
        or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")
    )
    if key and res and os.environ.get("OPENAI_DEPLOYMENT"):
        return "azure_openai"
    if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") and os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
        return "foundry"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


if __name__ == "__main__":
    # never print secret values, only whether they are set
    for k in (
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "CLAUDE_DEPLOYMENT",
        "OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WHISPER_MODEL",
        "VOICE_MATCH_THRESHOLD",
    ):
        print(f"{k:30s} {'set' if os.environ.get(k) else 'EMPTY'}")
    print(f"\nLLM backend: {llm_backend()}")

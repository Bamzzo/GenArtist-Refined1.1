"""
Unified configuration center for GenArtist.
Loads from .env via python-dotenv; exposes API key and pathlib-based paths.
Creates MODEL_ZOO_DIR and WORK_DIR if they do not exist.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (directory containing this config.py)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# API key (empty string if unset)
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# OpenAI-compatible base URL (for proxies or local gateways)
# 必须由 .env 或环境变量设置，不写死 openai.com
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip()

# Paths: always absolute; default to project root relative when env unset
_PROJECT_ROOT = Path(__file__).resolve().parent
_MODEL_ZOO_DIR = os.getenv("MODEL_ZOO_DIR", "").strip()
_WORK_DIR = os.getenv("WORK_DIR", "").strip()

MODEL_ZOO_DIR = Path(_MODEL_ZOO_DIR).resolve() if _MODEL_ZOO_DIR else _PROJECT_ROOT / "models"
WORK_DIR = Path(_WORK_DIR).resolve() if _WORK_DIR else _PROJECT_ROOT / "outputs"

# Ensure directories exist on import
MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

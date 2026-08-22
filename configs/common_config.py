"""Common variables — sourced from .env / environment; no hardcoded model fallbacks."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from configs.prompts import *


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip() or default


class CommonConfig:
    def __init__(self) -> None:
        llm_url = _env("LLM_BASE_URL")
        llm_key = _env("LLM_API_KEY")
        llm_model = _env("LLM_MODEL_NAME")

        self.__dict__ = {
            "DEEPSEEK_CONFIG": {
                "model_name": llm_model,
                "url": llm_url,
                "authorization": llm_key,
            },
            "RFANNS_AGENT_CONFIG": {
                "model_name": _env("RFANNS_MODEL_NAME", llm_model),
                "url": llm_url,
                "authorization": llm_key,
                "max_tokens": int(_env("RFANNS_MAX_TOKENS", "4096")),
                "temperature": float(_env("RFANNS_TEMPERATURE", "0.2")),
            },
            "HNSWLIB_AGENT_CONFIG": {
                "model_name": _env("HNSWLIB_MODEL_NAME", llm_model),
                "url": llm_url,
                "authorization": llm_key,
                "max_tokens": int(_env("HNSWLIB_MAX_TOKENS", "4096")),
                "temperature": float(_env("HNSWLIB_TEMPERATURE", "0.2")),
            },
            "SANDBOX": {
                "tool_link": "<address>:30008"
            },
            "SOLVER_PROMPT": {
                "user_prompt": SolverPrompt_User_Template,
                "assistant_prefix": SolverPrompt_Assistant_Template
            },
            "CRITIC_PROMPT": {
                "user_prompt": CriticPrompt_User_Template,
                "assistant_prefix": CriticPrompt_Assistant_Template
            },
            "CRITIC_WITH_SUGGESTION_PROMPT": {
                "user_prompt": CriticWithSuggestionPrompt_User_Template,
            },
            "REFINE_PROMPT": {
                "user_prompt": RefinePrompt_User_Template,
                "assistant_prefix": RefinePrompt_Assistant_Template
            },
            "QUALITY_PROMPT": {
                "user_prompt": QualityPrompt_User_Template,
            },
            "SELECTOR_PROMPT": {
                "user_prompt": SelectPrompt_User_Template,
                "assistant_prefix": SelectPrompt_Assistant_Template
            },
            "RFANNS_PROPOSER_PROMPT": {
                "user_prompt": RFANNSProposerPrompt_User_Template,
                "assistant_prefix": RFANNSProposerPrompt_Assistant_Template,
            },
            "RFANNS_REFLECTOR_PROMPT": {
                "user_prompt": RFANNSReflectorPrompt_User_Template,
                "assistant_prefix": RFANNSReflectorPrompt_Assistant_Template,
            },
        }

    def __getitem__(self, key):
        return self.__dict__.get(key, None)

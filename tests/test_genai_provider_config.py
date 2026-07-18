import utils.genai_handler as genai_handler


def test_provider_order_normalizes_and_deduplicates(monkeypatch):
    monkeypatch.setenv("AGRIBLOOM_LLM_ORDER", "openai,unknown,nvidia,openai,ollama")

    assert genai_handler._provider_order() == ["openai", "nvidia", "ollama"]


def test_openai_provider_requires_key_and_model(monkeypatch):
    monkeypatch.setattr(genai_handler, "_NVIDIA_API_KEYS", [])
    monkeypatch.setattr(genai_handler, "API_KEYS", [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)

    assert "openai" not in genai_handler.get_configured_llm_providers()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert "openai" not in genai_handler.get_configured_llm_providers()

    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert "openai" in genai_handler.get_configured_llm_providers()


def test_openai_compatible_provider_requires_base_url(monkeypatch):
    monkeypatch.setattr(genai_handler, "_NVIDIA_API_KEYS", [])
    monkeypatch.setattr(genai_handler, "API_KEYS", [])
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GROK_MODEL", raising=False)
    monkeypatch.delenv("GROK_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    assert "openai_compatible" not in genai_handler.get_configured_llm_providers()

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "test-model")
    assert "openai_compatible" not in genai_handler.get_configured_llm_providers()

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://example.test/v1/chat/completions")
    assert "openai_compatible" in genai_handler.get_configured_llm_providers()

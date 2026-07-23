import pytest

from app.services.llm import _is_transient_ollama_error


def test_transient_ollama_errors_are_detected():
    assert _is_transient_ollama_error(TimeoutError("read timed out"))
    assert _is_transient_ollama_error(ConnectionError("HTTPConnectionPool"))
    assert _is_transient_ollama_error(RuntimeError("Connection aborted"))
    assert not _is_transient_ollama_error(RuntimeError("model not found"))

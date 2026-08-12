import pytest
from cochem_knowledge_sync import semantic_chunker

def test_semantic_chunker() -> None:
    content = "# Section 1\nThis is a test block of text with enough length to pass the threshold.\n```python\n# This is a comment inside code\nx = 1\n```\n# Section 2\nAnother section with tags #troubleshooting."
    chunks = semantic_chunker(content, "test.md")
    assert len(chunks) >= 1
    assert "id" in chunks[0]
    assert "metadata" in chunks[0]

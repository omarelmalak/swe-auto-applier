"""Unit tests for FAISS knowledge-base search (no live Cohere calls)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.knowledge_base import (
    KnowledgeBase,
    chunk_resume_text,
    extract_resume_text,
)


DIM = 32


def _hash_embed(
    texts: list[str] | tuple[str, ...],
    *,
    input_type: str = "search_document",
) -> list[list[float]]:
    """
    Deterministic pseudo-embeddings for tests.

    Similar texts that share tokens get closer vectors via bag-of-hashed-tokens,
    so semantic search ranking is meaningful without an API key.
    `input_type` mirrors the Cohere embedder signature and is ignored here.
    """
    _ = input_type
    vectors: list[list[float]] = []
    for text in texts:
        vec = np.zeros(DIM, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Mix several bytes into several dimensions so collisions are softer.
            for i in range(0, 16, 2):
                idx = digest[i] % DIM
                sign = 1.0 if digest[i + 1] % 2 == 0 else -1.0
                vec[idx] += sign
        vectors.append(vec.tolist())
    return vectors


@pytest.fixture
def kb_paths(tmp_path: Path) -> dict[str, Path]:
    resume_pdf = Path(__file__).resolve().parents[1] / "data" / "resume.pdf"
    qa_path = tmp_path / "qa_bank.json"
    qa_path.write_text("[]\n", encoding="utf-8")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    return {
        "resume_pdf_path": resume_pdf,
        "qa_bank_path": qa_path,
        "index_path": index_dir / "index.faiss",
        "metadata_path": index_dir / "metadata.json",
        "preview_path": index_dir / "chunk_preview.html",
    }


@pytest.fixture
def built_kb(kb_paths: dict[str, Path]) -> KnowledgeBase:
    kb = KnowledgeBase(
        embed_fn=_hash_embed,
        dim=DIM,
        **kb_paths,
    )
    kb.rebuild()
    return kb


def test_extract_resume_text_not_empty() -> None:
    resume_pdf = Path(__file__).resolve().parents[1] / "data" / "resume.pdf"
    text = extract_resume_text(resume_pdf)
    assert len(text) > 50
    # Soft identity check — name should appear if PDF text layer is intact.
    assert "Omar" in text or "El Malak" in text or "@" in text


def test_chunk_resume_text_splits_sections() -> None:
    text = (
        "Omar El Malak\nSoftware Engineer\n"
        "Skills\nPython, TypeScript, SQL\n"
        "Education\nB.S. Computer Science\n"
        "Experience\n"
        "Backend Intern May 2026 – August 2026\nHubSpot\nBuilt APIs.\n"
        "Frontend Intern September 2025 – December 2025\nScotiabank\nBuilt UI."
    )
    chunks = chunk_resume_text(text)
    assert len(chunks) >= 4  # contact, skills, education, 2 roles (at least)
    joined = "\n".join(chunks).lower()
    assert "education" in joined
    assert "skills" in joined


def test_chunk_keeps_role_bullets_together() -> None:
    """A job's bullets must live in the same chunk as its title/dates."""
    resume_pdf = Path(__file__).resolve().parents[1] / "data" / "resume.pdf"
    text = extract_resume_text(resume_pdf)
    chunks = chunk_resume_text(text)

    scotiabank = [c for c in chunks if "Scotiabank" in c]
    assert len(scotiabank) == 1
    role = scotiabank[0]
    assert "Frontend Engineering Intern" in role
    assert "September 2025" in role
    assert "multilanguage formatting library" in role
    assert "year-end showcase" in role
    # Must not appear again elsewhere (no split leftover).
    assert sum("Scotiabank" in c for c in chunks) == 1


def test_chunk_contact_is_separate_from_experience() -> None:
    resume_pdf = Path(__file__).resolve().parents[1] / "data" / "resume.pdf"
    text = extract_resume_text(resume_pdf)
    chunks = chunk_resume_text(text)
    contact = chunks[0]
    assert "@" in contact or "gmail" in contact.lower()
    assert "Scotiabank" not in contact
    assert "HubSpot" not in contact


def test_rebuild_indexes_resume_chunks(built_kb: KnowledgeBase) -> None:
    assert built_kb.index is not None
    assert built_kb.index.ntotal == len(built_kb.chunks)
    assert built_kb.index.ntotal > 0
    assert all(c.source == "resume" for c in built_kb.chunks)
    assert built_kb.index_path.exists()
    assert built_kb.metadata_path.exists()
    assert built_kb.preview_path.exists()
    html = built_kb.preview_path.read_text(encoding="utf-8")
    assert "Chunk preview" in html
    assert 'class="chunk"' in html
    assert f"#{built_kb.chunks[0].id}" in html


def test_search_returns_ranked_results(built_kb: KnowledgeBase) -> None:
    hits = built_kb.search("software engineering experience python", k=3)
    assert 1 <= len(hits) <= 3
    # Scores should be descending (IndexFlatIP + normalized vectors).
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(h.chunk.text for h in hits)


def test_search_empty_query_returns_empty(built_kb: KnowledgeBase) -> None:
    assert built_kb.search("   ") == []


def test_search_prefers_relevant_chunk(tmp_path: Path) -> None:
    """Exact token overlap should rank the matching document highest."""
    qa_path = tmp_path / "qa_bank.json"
    qa_path.write_text("[]\n", encoding="utf-8")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()

    # Bypass PDF: build a tiny synthetic index by monkeypatching extract/chunk.
    kb = KnowledgeBase(
        embed_fn=_hash_embed,
        dim=DIM,
        resume_pdf_path=tmp_path / "unused.pdf",
        qa_bank_path=qa_path,
        index_path=index_dir / "index.faiss",
        metadata_path=index_dir / "metadata.json",
    )

    docs = [
        "Candidate email address is omar@example.com",
        "Graduated with a degree in biology and chemistry",
        "Work authorization requires sponsorship discussion",
    ]
    matrix = np.asarray(_hash_embed(docs), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    import faiss
    from core.knowledge_base import DocumentChunk

    index = faiss.IndexFlatIP(DIM)
    index.add(matrix)
    kb.index = index
    kb.chunks = [
        DocumentChunk(id=i, source="resume", text=doc) for i, doc in enumerate(docs)
    ]

    hits = kb.search("what is the candidate email address", k=3)
    assert hits[0].chunk.text == docs[0]
    assert hits[0].score >= hits[1].score


def test_add_qa_is_retrievable(built_kb: KnowledgeBase, kb_paths: dict[str, Path]) -> None:
    built_kb.add_qa(
        question="Are you authorized to work in the United States?",
        answer="Yes, I am a U.S. citizen and do not require sponsorship.",
    )

    qa_entries = json.loads(kb_paths["qa_bank_path"].read_text(encoding="utf-8"))
    assert len(qa_entries) == 1
    assert "sponsorship" in qa_entries[0]["answer"].lower()

    hits = built_kb.search("work authorization sponsorship united states", k=3)
    assert any(h.chunk.source == "qa" for h in hits)
    qa_hit = next(h for h in hits if h.chunk.source == "qa")
    assert "U.S. citizen" in qa_hit.chunk.answer or "sponsorship" in qa_hit.chunk.text


def test_load_persisted_index(built_kb: KnowledgeBase, kb_paths: dict[str, Path]) -> None:
    reloaded = KnowledgeBase(
        embed_fn=_hash_embed,
        dim=DIM,
        **kb_paths,
    )
    reloaded.load()
    assert reloaded.index is not None
    assert reloaded.index.ntotal == built_kb.index.ntotal
    hits = reloaded.search("skills education", k=2)
    assert len(hits) == 2

"""Local FAISS knowledge base for resume chunks and learned Q&A pairs."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import cohere
import faiss
import numpy as np
import pdfplumber

import config

# texts, input_type -> embeddings  (Cohere needs search_document vs search_query)
EmbedFn = Callable[..., list[list[float]]]

# Cohere v3 embed endpoints typically cap batch size around 96 texts.
_COHERE_EMBED_BATCH_SIZE = 96


@dataclass
class DocumentChunk:
    """One searchable unit stored alongside a FAISS vector."""

    id: int
    source: str  # "resume" | "qa"
    text: str
    question: str | None = None
    answer: str | None = None


@dataclass
class SearchResult:
    """A ranked retrieval hit from the knowledge base."""

    chunk: DocumentChunk
    score: float


def extract_resume_text(pdf_path: Path) -> str:
    """Extract plain text from a resume PDF via pdfplumber."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(text)

    combined = "\n\n".join(pages).strip()
    if not combined:
        raise ValueError(f"No extractable text found in {pdf_path}")
    return combined


_SECTION_HEADER_RE = re.compile(
    r"^(skills|education|experience|projects|work experience|"
    r"technical skills|certifications|awards)\s*$",
    re.IGNORECASE,
)

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
)
# e.g. "Backend Engineering Intern May 2026 – August 2026"
_ROLE_START_RE = re.compile(
    rf".*{_MONTH}\s+\d{{4}}\s*[–\-]\s*(?:Present|{_MONTH}\s+\d{{4}})\s*$",
    re.IGNORECASE,
)

# Hard ceiling only for pathological blobs; role chunks may exceed the soft target
# so a job's title/dates/bullets stay in one vector.
_HARD_MAX_CHARS = 2500


def chunk_resume_text(text: str, max_chars: int = 900) -> list[str]:
    """
    Section/role-aware resume chunking.

    Keeps contact, Skills, Education, and each Experience role as atomic units so
    Scotiabank (etc.) bullets are never split across two FAISS vectors. Char packing
    is only a fallback for unknown layouts / oversized blobs.
    """
    normalized = _normalize_resume_text(text)
    sections = _split_into_sections(normalized)
    if not sections:
        return _pack_blocks([normalized], max_chars=max_chars)

    chunks: list[str] = []
    for title, body in sections:
        body = body.strip()
        if not body and not title:
            continue

        if title.lower() in {"experience", "work experience"}:
            roles = _split_experience_roles(body)
            for role in roles:
                labeled = f"Experience\n{role}".strip()
                chunks.extend(_ensure_size(labeled, max_chars=max_chars))
            continue

        if title:
            labeled = f"{title}\n{body}".strip() if body else title
        else:
            labeled = body
        # Contact / skills / education: prefer one chunk; pack only if huge.
        chunks.extend(_ensure_size(labeled, max_chars=max_chars))

    return [c for c in chunks if c.strip()]


def _normalize_resume_text(text: str) -> str:
    """Clean PDF artifacts without destroying line structure used for role splits."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    raw_lines = [ln.strip() for ln in text.split("\n")]
    lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        stripped = raw_lines[i]
        # Drop decorative separator lines like "| | | |"
        if stripped and re.fullmatch(r"[|.\-–—\s]+", stripped):
            i += 1
            continue

        # pdfplumber often parks a lone "•" between wrapped bullet sentences.
        # Attach it to the *following* content line — but never to a section header,
        # or "•\nExperience" becomes "• Experience" and the section split breaks.
        if stripped == "•" and i + 1 < len(raw_lines):
            nxt = raw_lines[i + 1].strip()
            if nxt and not _SECTION_HEADER_RE.match(nxt):
                lines.append(f"• {nxt}")
                i += 2
                continue
            # orphan bullet before a header / EOF — drop it
            i += 1
            continue

        if stripped:
            lines.append(stripped)
        i += 1

    return "\n".join(lines).strip()


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_title_or_'', body), ...] in document order."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_title
        body = "\n".join(buf).strip()
        if body or current_title:
            sections.append((current_title, body))
        buf = []

    for line in lines:
        if _SECTION_HEADER_RE.match(line):
            flush()
            current_title = line.strip()
            continue
        buf.append(line)
    flush()
    return sections


def _split_experience_roles(body: str) -> list[str]:
    """
    Split an Experience section into one string per job.

    A new role starts on any line that contains an employment date range
    (title + dates are usually on the same line in this resume format).
    """
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if not lines:
        return []

    roles: list[str] = []
    current: list[str] = []

    for line in lines:
        if _ROLE_START_RE.match(line) and current:
            roles.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        roles.append("\n".join(current).strip())
    return roles


def _ensure_size(text: str, max_chars: int) -> list[str]:
    """Keep atomic when possible; only pack-split past the hard ceiling."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max(_HARD_MAX_CHARS, max_chars):
        return [text]
    return _pack_blocks([text], max_chars=max_chars)


def _pack_blocks(blocks: list[str], max_chars: int) -> list[str]:
    """Fallback char packer for non-section text / oversized blobs."""
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        for piece in _split_long_block(block, max_chars):
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = piece
    flush()
    return chunks


def _split_long_block(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]

    lines = block.split("\n")
    pieces: list[str] = []
    buf = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        candidate = f"{buf}\n{line}".strip() if buf else line
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
        if len(line) <= max_chars:
            buf = line
        else:
            for i in range(0, len(line), max_chars):
                pieces.append(line[i : i + max_chars])
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def load_qa_bank(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"QA bank must be a JSON list: {path}")
    return data


def save_qa_bank(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
        f.write("\n")


_PREVIEW_COLORS = [
    "#2563eb",  # blue
    "#059669",  # green
    "#d97706",  # amber
    "#db2777",  # pink
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#dc2626",  # red
    "#4f46e5",  # indigo
]


def write_chunk_preview(
    chunks: Sequence[DocumentChunk],
    path: Path,
    *,
    resume_name: str = "resume.pdf",
) -> Path:
    """
    Write a simple HTML board showing each indexed chunk in its own colored box.

    Drawn on the *extracted text* (not PDF page coordinates). Overlaying boxes on
    the original PDF is fragile because pdfplumber line breaks rarely match layout
    boxes 1:1 — this preview is the reliable way to inspect chunk boundaries after
    every rebuild/upload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for chunk in chunks:
        color = _PREVIEW_COLORS[chunk.id % len(_PREVIEW_COLORS)]
        label = chunk.source.upper()
        if chunk.source == "resume":
            first_line = chunk.text.split("\n", 1)[0].strip()
            if first_line:
                label = first_line[:60]
        title = html.escape(f"#{chunk.id} · {label}")
        body = html.escape(chunk.text)
        cards.append(
            f"""
            <article class="chunk" style="--accent:{color}">
              <header>
                <span class="badge">{html.escape(chunk.source)}</span>
                <h2>{title}</h2>
                <span class="meta">{len(chunk.text)} chars · ~{len(chunk.text.split())} words</span>
              </header>
              <pre>{body}</pre>
            </article>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Resume chunk preview — {html.escape(resume_name)}</title>
  <style>
    :root {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: #0f172a;
      background: #f1f5f9;
    }}
    body {{
      margin: 0 auto;
      max-width: 920px;
      padding: 2rem 1.25rem 4rem;
    }}
    h1 {{
      font-size: 1.5rem;
      margin: 0 0 0.35rem;
    }}
    .sub {{
      color: #64748b;
      margin-bottom: 1.75rem;
    }}
    .chunk {{
      background: #fff;
      border: 3px solid var(--accent);
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      margin-bottom: 1.25rem;
      overflow: hidden;
    }}
    .chunk header {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      align-items: baseline;
      padding: 0.85rem 1rem;
      background: color-mix(in srgb, var(--accent) 12%, white);
      border-bottom: 2px solid var(--accent);
    }}
    .chunk h2 {{
      font-size: 1rem;
      margin: 0;
      flex: 1;
    }}
    .badge {{
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #fff;
      background: var(--accent);
      padding: 0.15rem 0.45rem;
      border-radius: 999px;
    }}
    .meta {{
      color: #64748b;
      font-size: 0.8rem;
    }}
    pre {{
      margin: 0;
      padding: 1rem 1.1rem 1.25rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.86rem;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <h1>Chunk preview</h1>
  <p class="sub">
    Source: <strong>{html.escape(resume_name)}</strong> ·
    {len(chunks)} chunk{"s" if len(chunks) != 1 else ""} ·
    regenerated on every <code>rebuild</code>
  </p>
  {"".join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def make_cohere_embedder(
    api_key: str | None = None,
    model: str | None = None,
) -> EmbedFn:
    """Return an embedding function backed by Cohere (document/query input types)."""
    key = api_key if api_key is not None else config.COHERE_API_KEY
    if not key:
        raise ValueError(
            "COHERE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = cohere.ClientV2(api_key=key)
    model_name = model or config.COHERE_EMBEDDING_MODEL

    def _embed(
        texts: Sequence[str],
        *,
        input_type: str = "search_document",
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        batch = list(texts)
        for start in range(0, len(batch), _COHERE_EMBED_BATCH_SIZE):
            chunk = batch[start : start + _COHERE_EMBED_BATCH_SIZE]
            response = client.embed(
                texts=chunk,
                model=model_name,
                input_type=input_type,
                embedding_types=["float"],
            )
            float_embeddings = response.embeddings.float
            if float_embeddings is None:
                raise RuntimeError("Cohere embed response missing float embeddings.")
            embeddings.extend(float_embeddings)
        return embeddings

    return _embed


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class KnowledgeBase:
    """FAISS-backed semantic search over resume chunks and Q&A pairs."""

    def __init__(
        self,
        *,
        embed_fn: EmbedFn | None = None,
        dim: int | None = None,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        qa_bank_path: Path | None = None,
        resume_pdf_path: Path | None = None,
        preview_path: Path | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.dim = dim or config.EMBEDDING_DIM
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or config.FAISS_METADATA_PATH
        self.qa_bank_path = qa_bank_path or config.QA_BANK_PATH
        self.resume_pdf_path = resume_pdf_path or config.RESUME_PDF_PATH
        self.preview_path = preview_path or config.CHUNK_PREVIEW_PATH

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []

    def _require_embed_fn(self) -> EmbedFn:
        if self.embed_fn is None:
            self.embed_fn = make_cohere_embedder()
        return self.embed_fn

    def _embed_matrix(
        self,
        texts: Sequence[str],
        *,
        input_type: str = "search_document",
    ) -> np.ndarray:
        vectors = self._require_embed_fn()(texts, input_type=input_type)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dim}), got {arr.shape}"
            )
        return _l2_normalize(arr)

    def rebuild(self) -> list[DocumentChunk]:
        """Rebuild the entire index from the resume PDF and qa_bank.json."""
        resume_text = extract_resume_text(self.resume_pdf_path)
        resume_chunks = chunk_resume_text(resume_text)
        qa_entries = load_qa_bank(self.qa_bank_path)

        chunks: list[DocumentChunk] = []
        texts: list[str] = []

        for text in resume_chunks:
            chunk_id = len(chunks)
            chunks.append(DocumentChunk(id=chunk_id, source="resume", text=text))
            texts.append(text)

        for entry in qa_entries:
            question = str(entry.get("question", "")).strip()
            answer = str(entry.get("answer", "")).strip()
            if not question or not answer:
                continue
            text = f"Q: {question}\nA: {answer}"
            chunk_id = len(chunks)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source="qa",
                    text=text,
                    question=question,
                    answer=answer,
                )
            )
            texts.append(text)

        if not chunks:
            raise ValueError("Nothing to index: resume produced no chunks and QA bank is empty.")

        matrix = self._embed_matrix(texts, input_type="search_document")
        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)

        self.index = index
        self.chunks = chunks
        self.save()
        write_chunk_preview(
            chunks,
            self.preview_path,
            resume_name=self.resume_pdf_path.name,
        )
        return chunks

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Call rebuild() first.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "dim": self.dim,
            "chunks": [asdict(c) for c in self.chunks],
        }
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS artifacts. Expected {self.index_path} and "
                f"{self.metadata_path}. Run rebuild first."
            )
        self.index = faiss.read_index(str(self.index_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.dim = int(payload.get("dim", self.dim))
        self.chunks = [DocumentChunk(**item) for item in payload["chunks"]]

    def ensure_loaded(self) -> None:
        if self.index is None:
            self.load()

    def search(
        self,
        query: str,
        k: int = 3,
        *,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """
        Return the top-k most similar chunks for a natural-language query.

        Default k=3 so a small index does not dump the entire resume into the
        LLM context. Optional min_score drops weak cosine matches.
        """
        query = query.strip()
        if not query:
            return []

        self.ensure_loaded()
        assert self.index is not None

        if self.index.ntotal == 0:
            return []

        k = max(1, min(k, self.index.ntotal))
        query_vec = self._embed_matrix([query], input_type="search_query")
        scores, indices = self.index.search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            score_f = float(score)
            if min_score is not None and score_f < min_score:
                continue
            results.append(
                SearchResult(chunk=self.chunks[int(idx)], score=score_f)
            )
        return results

    def add_qa(self, question: str, answer: str) -> DocumentChunk:
        """
        Persist a learned Q&A pair and append it to the live FAISS index.

        Used by the Discord correction loop in Phase 2.
        """
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            raise ValueError("Both question and answer are required.")

        self.ensure_loaded()
        assert self.index is not None

        entries = load_qa_bank(self.qa_bank_path)
        entries.append({"question": question, "answer": answer})
        save_qa_bank(self.qa_bank_path, entries)

        text = f"Q: {question}\nA: {answer}"
        chunk = DocumentChunk(
            id=len(self.chunks),
            source="qa",
            text=text,
            question=question,
            answer=answer,
        )
        vector = self._embed_matrix([text], input_type="search_document")
        self.index.add(vector)
        self.chunks.append(chunk)
        self.save()
        return chunk


def rebuild_default() -> list[DocumentChunk]:
    """Rebuild the project-default knowledge base using Cohere embeddings."""
    kb = KnowledgeBase()
    return kb.rebuild()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage the AutoApply FAISS knowledge base.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("rebuild", help="Rebuild index from resume.pdf + qa_bank.json")
    sub.add_parser(
        "preview",
        help="Regenerate chunk_preview.html from the saved index (no re-embed)",
    )

    search_parser = sub.add_parser("search", help="Run a semantic search query")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("-k", type=int, default=3)
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop hits below this cosine similarity (e.g. 0.35)",
    )

    args = parser.parse_args(argv)

    if args.command == "rebuild":
        chunks = rebuild_default()
        resume_n = sum(1 for c in chunks if c.source == "resume")
        qa_n = sum(1 for c in chunks if c.source == "qa")
        print(f"Indexed {len(chunks)} chunks ({resume_n} resume, {qa_n} qa).")
        print(f"Wrote {config.FAISS_INDEX_PATH}")
        print(f"Preview {config.CHUNK_PREVIEW_PATH}")
        print(f"Open with: open {config.CHUNK_PREVIEW_PATH}")
        return

    if args.command == "preview":
        kb = KnowledgeBase()
        kb.load()
        path = write_chunk_preview(
            kb.chunks,
            kb.preview_path,
            resume_name=kb.resume_pdf_path.name,
        )
        print(f"Preview {path}")
        print(f"Open with: open {path}")
        return

    if args.command == "search":
        kb = KnowledgeBase()
        kb.load()
        hits = kb.search(args.query, k=args.k, min_score=args.min_score)
        if not hits:
            print("No results.")
            return
        for i, hit in enumerate(hits, start=1):
            print(
                f"{i}. [{hit.chunk.source}] id={hit.chunk.id} "
                f"score={hit.score:.4f} chars={len(hit.chunk.text)}"
            )
            preview = hit.chunk.text.replace("\n", " | ")
            if len(preview) > 240:
                preview = preview[:237] + "..."
            print(f"   {preview}")


if __name__ == "__main__":
    main()

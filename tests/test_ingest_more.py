"""
Ingestion's other paths: reading a PDF, a document that does not exist, and a
failure half-way through, which must leave the document marked "failed" (so the
client sees it) and must not leave partial chunks behind.
"""
import pytest

from app import ingest
from app.config import settings
from app.models import Document


def _minimal_pdf(text: str) -> bytes:
    """A one-page PDF containing `text`, built by hand so no PDF library is needed."""
    stream = f"BT /F1 18 Tf 20 70 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 144]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj".encode() + body + b"endobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_reads_text_out_of_a_pdf():
    assert "Hello Sift" in ingest._read_text("doc.pdf", _minimal_pdf("Hello Sift"))


def test_reads_plain_text_and_ignores_bad_bytes():
    assert ingest._read_text("notes.txt", b"ok \xff\xfe fine") == "ok  fine"


def test_a_missing_document_is_an_error(db_session):
    with pytest.raises(ValueError):
        ingest.ingest_into(db_session, 123456)


def test_a_failure_marks_the_document_failed_and_leaves_no_chunks(db_session, tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_embed(text):
        calls["n"] += 1
        if calls["n"] == 3:                         # fail part-way through the chunks
            raise RuntimeError("embedding service unavailable")
        return [0.0] * settings.embed_dim

    monkeypatch.setattr(ingest, "embed", flaky_embed)
    path = tmp_path / "doc.txt"
    path.write_text("sentence. " * 300)
    doc = Document(filename="doc.txt", status="pending", source_path=str(path))
    db_session.add(doc)
    db_session.commit()

    with pytest.raises(RuntimeError):
        ingest.ingest_into(db_session, doc.id)

    db_session.refresh(doc)
    assert doc.status == "failed"                   # visible to the client polling the status
    assert ingest.chunk_count(db_session, doc.id) == 0   # the two chunks before the failure rolled back

"""The chunker is a pure function — testable without any services."""
from app import ingest
from app.config import settings


def test_chunks_cover_the_text():
    text = "word " * 500                      # ~2500 chars
    chunks = ingest._chunk(text)
    assert len(chunks) > 1
    # every chunk is non-empty and within the size limit
    assert all(0 < len(c) <= settings.chunk_size for c in chunks)


def test_overlap_shares_text_between_neighbours():
    # distinctive, long text so neighbouring chunks visibly overlap
    text = "".join(f"[{i:04d}]" for i in range(400))   # 2400 chars, no spaces
    chunks = ingest._chunk(text)
    assert len(chunks) >= 2
    # the end of chunk 0 should reappear at the start of chunk 1 (overlap region)
    tail = chunks[0][-settings.chunk_overlap:]
    assert tail and tail in chunks[1]


def test_short_text_is_one_chunk():
    assert ingest._chunk("just a little text") == ["just a little text"]


def test_empty_text_yields_no_chunks():
    assert ingest._chunk("   ") == []

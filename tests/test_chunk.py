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


def test_no_chunk_starts_or_ends_inside_a_word():
    # Words of uneven length, so window edges land mid-word; "<" and ">" mark
    # both ends, so any cut piece ("<12", "3>") is not in the set.
    words = [f"<{i}>" for i in range(700)]
    chunks = ingest._chunk(" ".join(words))
    whole = set(words)
    assert len(chunks) > 3
    for c in chunks:
        parts = c.split()
        assert parts[0] in whole and parts[-1] in whole


def test_neighbours_still_overlap_after_snapping_to_words():
    chunks = ingest._chunk(" ".join(f"<{i}>" for i in range(700)))
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()                # a's last word reappears in b


def test_whitespace_runs_become_single_spaces():
    assert ingest._chunk("line one\n\n  line   two\t") == ["line one line two"]

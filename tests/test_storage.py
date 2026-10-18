"""
Upload storage. The filename comes from the client, so it is untrusted input:
a name like ``../../etc/passwd`` must not be able to write outside the upload
folder.
"""
import os

from app import storage
from app.config import settings


def test_safe_name_strips_path_separators():
    name = storage._safe_name("../../etc/passwd")
    assert "/" not in name and "\\" not in name
    assert name == ".._.._etc_passwd"


def test_safe_name_keeps_ordinary_names_readable():
    assert storage._safe_name("report-2024_v2.pdf") == "report-2024_v2.pdf"
    assert storage._safe_name("my notes (1).txt") == "my_notes__1_.txt"


def test_safe_name_never_returns_empty():
    assert storage._safe_name("") == "file"


def test_a_hostile_name_is_saved_inside_the_upload_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path))
    path = storage.save_upload(7, "../../outside.txt", b"data")
    assert os.path.commonpath([os.path.realpath(path), str(tmp_path.resolve())]) == str(tmp_path.resolve())
    assert os.path.basename(path).startswith("7_")


def test_save_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path / "new-folder"))   # created on demand
    path = storage.save_upload(1, "a.bin", bytes(range(256)))
    assert storage.read_upload(path) == bytes(range(256))

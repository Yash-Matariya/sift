"""
File storage for uploads.

The upload endpoint saves the raw file here so the background worker — a separate
process — can read it later. Locally this is a directory; in Docker it's a volume
shared between the API and the worker; in production it would be object storage
like S3. Keeping it behind these two functions means only this file changes if
the backing store changes.
"""
import os
import re

from app.config import settings


def _safe_name(filename: str) -> str:
    """Strip anything but simple filename characters, to avoid path traversal."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "file"


def save_upload(doc_id: int, filename: str, data: bytes) -> str:
    """Save the raw bytes and return the path they were written to."""
    os.makedirs(settings.uploads_dir, exist_ok=True)
    path = os.path.join(settings.uploads_dir, f"{doc_id}_{_safe_name(filename)}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def read_upload(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def delete_upload(path: str | None) -> None:
    """Remove a saved upload. A file that is already gone is fine: deleting is the goal."""
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

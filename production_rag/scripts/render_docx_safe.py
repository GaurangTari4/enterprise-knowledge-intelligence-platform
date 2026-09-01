from __future__ import annotations

import runpy
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
import uuid


class _NoCleanupTempDir(AbstractContextManager[str]):
    def __init__(self, prefix: str = "tmp_", dir: str | None = None):
        base = Path(dir or tempfile.gettempdir())
        self._path = str(base / f"{prefix}{uuid.uuid4().hex}")
        Path(self._path).mkdir(parents=True, exist_ok=True)
        (Path(self._path) / "xdg_config").mkdir(parents=True, exist_ok=True)
        (Path(self._path) / "xdg_cache").mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> str:
        return self._path

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def main() -> None:
    tempfile._resetperms = lambda path: None  # type: ignore[attr-defined]
    tempfile.TemporaryDirectory = _NoCleanupTempDir  # type: ignore[assignment]
    runpy.run_path(
        renderer,
        run_name="__main__",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import threading
from pathlib import Path


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def publish(self, result: object) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else result
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

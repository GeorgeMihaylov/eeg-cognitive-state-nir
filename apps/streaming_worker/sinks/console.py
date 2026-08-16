from __future__ import annotations

import json


class ConsoleSink:
    def publish(self, result: object) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else result
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def close(self) -> None:
        return None

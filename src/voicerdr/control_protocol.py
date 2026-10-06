import json
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class ControlRequest:
    id: str
    method: str
    params: dict[str, Any]

    @classmethod
    def parse(cls, line: str) -> Self:
        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key}")
                value[key] = item
            return value

        data = json.loads(line, object_pairs_hook=no_duplicates)
        if not isinstance(data, dict):
            raise TypeError("request must be a JSON object")
        method = data.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("missing method")
        req_id = data.get("id")
        if req_id is None:
            req_id = "0"
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError("params must be an object")
        return cls(id=str(req_id), method=method, params=params)


def ok(req_id: str, result: dict[str, Any] | None = None) -> str:
    return json.dumps({"id": req_id, "ok": True, "result": result or {}}) + "\n"


def err(req_id: str, message: str, code: str = "error") -> str:
    return (
        json.dumps(
            {
                "id": req_id,
                "ok": False,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    )

"""Generate docs/API.md from the FastAPI OpenAPI schema plus hand-written notes.

Run from the repo root:  cd backend && VIRTUAL_ENV= uv run python ../scripts/gen_api_md.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "doc")
os.environ.setdefault("LEGALRAG_SKIP_WARMUP", "1")

from legalrag.api.app import create_app  # noqa: E402

NOTES = """
## Ghi chú

- Base URL: `http://localhost:8000/api/v1`. Tài liệu tương tác: `/api/v1/docs`.
- `POST /chat` trả về **SSE** (`text/event-stream`) với các event `status`, `sources`, `token`,
  `citations`, `done`, `error` (xem DESIGN §3.9). `POST /chat/sync` trả JSON một lần.
- Mọi lỗi trả JSON `{code, message, request_id}` với `message` tiếng Việt; header `X-Request-ID`
  được sinh tự động nếu client không gửi.
- `/admin/*` yêu cầu header `X-Admin-Token` = `ADMIN_TOKEN`. `/chat*` giới hạn 30 yêu cầu/phút/IP
  (Redis; tự chuyển sang bộ đếm trong tiến trình nếu Redis không khả dụng).
- `/health` và `/health/ready` trả 503 khi Qdrant hoặc Postgres không sẵn sàng hoặc khi API chưa
  nạp xong mô hình; `/health/live` luôn 200 khi tiến trình còn sống.
"""


def main() -> None:
    app = create_app()
    spec = app.openapi()
    lines = ["# API Reference", "", f"_Sinh tự động từ OpenAPI ({spec['info']['version']})._", ""]
    lines += ["| Method | Path | Mô tả |", "|---|---|---|"]
    for path, methods in sorted(spec["paths"].items()):
        for method, op in methods.items():
            summary = op.get("summary") or op.get("operationId", "")
            lines.append(f"| {method.upper()} | `{path}` | {summary} |")
    lines.append("")
    for path, methods in sorted(spec["paths"].items()):
        for method, op in methods.items():
            lines.append(f"### {method.upper()} `{path}`")
            lines.append("")
            params = op.get("parameters", [])
            if params:
                lines.append("Tham số:")
                for p in params:
                    req = " (bắt buộc)" if p.get("required") else ""
                    lines.append(f"- `{p['name']}` ({p['in']}){req}")
                lines.append("")
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {})
            if body:
                ref = body.get("schema", {}).get("$ref", "")
                lines.append(f"Body: `{ref.rsplit('/', 1)[-1] or 'object'}`")
                lines.append("")
            codes = ", ".join(sorted(op.get("responses", {}).keys()))
            lines.append(f"Responses: {codes}")
            lines.append("")
    out = Path(__file__).resolve().parents[1] / "docs" / "API.md"
    out.write_text("\n".join(lines) + NOTES, encoding="utf-8")
    print(f"wrote {out} ({len(spec['paths'])} paths)")


if __name__ == "__main__":
    main()

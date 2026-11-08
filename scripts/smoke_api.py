"""Smoke-test a running API (spec §4.5): chat/sync, SSE, feedback, admin, search, validation, health.

Usage:  python -X utf8 scripts/smoke_api.py [--base http://localhost:8000/api/v1] [--token change-me]
Exit code 1 if any check fails. Uses only the standard library (urllib) so it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

OK: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (OK if cond else FAIL).append(f"{name} {detail}".strip())
    print(("✔ " if cond else "✘ ") + name + (f" — {detail}" if detail else ""))


def call(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: int = 300):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, (json.loads(raw) if raw.startswith("{") else raw), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw), dict(e.headers)
        except json.JSONDecodeError:
            return e.code, raw, dict(e.headers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/api/v1")
    ap.add_argument("--token", default="change-me")
    a = ap.parse_args()
    b = a.base

    st, body, _ = call("GET", f"{b}/health/ready")
    check("health/ready 200", st == 200, f"status={st} n_chunks={body.get('n_chunks') if isinstance(body, dict) else '?'}")

    st, body, hdr = call("POST", f"{b}/chat/sync", {"question": "Đối tượng nào không chịu thuế giá trị gia tăng?", "session_id": "smoke"}, {"X-Request-ID": "smoke-1"})
    ok = st == 200 and isinstance(body, dict) and body.get("grounded") is True and len(body.get("citations", {}).get("verified", [])) >= 1
    check("chat/sync grounded with verified citations", ok, f"status={st} cit={body.get('citations') if isinstance(body, dict) else body}")
    check("X-Request-ID echoed", hdr.get("X-Request-ID") == "smoke-1" or hdr.get("x-request-id") == "smoke-1")
    message_id = body.get("message_id") if isinstance(body, dict) else None
    if isinstance(body, dict):
        print("   sources:", [s["display_source"][:60] for s in body.get("sources", [])][:3])
        print("   answer:", body.get("answer_md", "")[:200].replace("\n", " "))

    st, body, _ = call("POST", f"{b}/chat/sync", {"question": "Hôm nay trời có mưa không?"})
    check("out_of_scope handled without grounding", st == 200 and isinstance(body, dict) and body.get("intent") == "out_of_scope" and body.get("grounded") is False, f"intent={body.get('intent') if isinstance(body, dict) else body}")

    st, body, _ = call("POST", f"{b}/chat/sync", {"question": "   "})
    check("blank question → 422 Vietnamese", st == 422 and "trống" in str(body))
    st, body, _ = call("POST", f"{b}/chat/sync", {"question": "a" * 2001})
    check("too long question → 422", st == 422 and "2000" in str(body))

    req = urllib.request.Request(f"{b}/chat", data=json.dumps({"question": "Chào bạn"}).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        sse = r.read().decode("utf-8")
    check("SSE stream has status/token/done events", all(f"event: {e}" in sse for e in ("status", "token", "done")), f"{len(sse)} bytes")

    if message_id:
        st, body, _ = call("POST", f"{b}/feedback", {"message_id": message_id, "rating": -1, "reason": "incomplete", "comment": "smoke"})
        check("feedback 201", st == 201, str(body)[:80])
    st, body, _ = call("POST", f"{b}/feedback", {"message_id": "nope", "rating": 1})
    check("feedback unknown message → 404", st == 404)

    st, body, _ = call("GET", f"{b}/admin/stats")
    check("admin without token → 401", st == 401)
    st, body, _ = call("GET", f"{b}/admin/stats", headers={"X-Admin-Token": a.token})
    neg = body.get("recent_negative_feedback", []) if isinstance(body, dict) else []
    check("admin/stats with token", st == 200 and isinstance(body, dict) and "collections" in body, f"collections={[(c['strategy'], c['n_chunks']) for c in body.get('collections', [])] if isinstance(body, dict) else body}")
    check("negative feedback visible in admin", any(f.get("message_id") == message_id for f in neg))

    st, body, _ = call("GET", f"{b}/search?q=thu%E1%BA%BF%20su%E1%BA%A5t%20gi%C3%A1%20tr%E1%BB%8B%20gia%20t%C4%83ng&top_k=3&mode=bm25")
    check("search bm25 returns results", st == 200 and isinstance(body, dict) and len(body.get("results", [])) == 3)
    if isinstance(body, dict) and body.get("results"):
        cid = body["results"][0]["chunk_id"]
        st, body, _ = call("GET", f"{b}/chunks/{urllib.request.quote(cid, safe='')}")
        check("chunks/{id} returns chunk with neighbours", st == 200 and isinstance(body, dict) and "chunk" in body)
        st, body, _ = call("GET", f"{b}/documents/{urllib.request.quote(body['chunk']['doc_id'], safe='')}" if isinstance(body, dict) else f"{b}/documents/x")
        check("documents/{id}", st == 200)

    st, body, _ = call("GET", f"{b}/config/public")
    check("config/public lists strategies", st == 200 and isinstance(body, dict) and body.get("strategies"), str(body.get("strategies") if isinstance(body, dict) else body))

    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# End-to-end demo: bring the stack up, wait for readiness, ask the smoke questions via the API.
set -euo pipefail
cd "$(dirname "$0")/.."
API=${API:-http://localhost:8000/api/v1}

if [ ! -f .env ]; then echo "Missing .env (cp .env.example .env and set ANTHROPIC_API_KEY)"; exit 1; fi
docker compose up -d --build
echo "Waiting for API readiness..."
for i in $(seq 1 90); do
  if curl -fs "$API/health/ready" >/dev/null 2>&1; then echo "API ready"; break; fi
  sleep 5
  [ "$i" = 90 ] && { echo "API not ready after 450s"; docker compose logs --tail=50 api; exit 1; }
done

ask() {
  printf '\n=== %s\n' "$1"
  curl -s "$API/chat/sync" -H 'Content-Type: application/json' \
    -d "$(printf '{"question": %s}' "$(printf '%s' "$1" | python -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")" \
    | python -c 'import json,sys; d=json.load(sys.stdin); print("grounded:",d["grounded"],"intent:",d["intent"]); print(d["answer_md"][:700]); print("sources:",[s["display_source"] for s in d["sources"]][:3])'
}
ask "Đối tượng nào không chịu thuế giá trị gia tăng?"
ask "Hôm nay trời có mưa không?"
ask "Luật Thuế GTGT quy định thuế suất 15% cho mặt hàng nào?"
echo
echo "Open http://localhost:3000 for the chat UI."

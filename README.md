# Vietnamese Legal QA — RAG (local, Docker)

Hệ thống hỏi đáp pháp luật Việt Nam theo kiến trúc RAG: người dùng hỏi bằng tiếng Việt → hệ thống truy xuất
các Điều/Khoản liên quan từ kho văn bản pháp luật (hybrid BM25 + dense bge-m3 → rerank) → Claude sinh câu trả lời
**chỉ dựa trên văn bản truy xuất**, **có trích dẫn `[#i]` được kiểm chứng**, và **từ chối khi không đủ căn cứ**.
Toàn bộ index, DB và UI chạy local bằng Docker; chỉ prompt + context đi tới Anthropic API.

Thiết kế chi tiết: [`docs/DESIGN.md`](docs/DESIGN.md) · Tiến độ: [`docs/PROGRESS.md`](docs/PROGRESS.md) ·
Quyết định kỹ thuật: [`docs/DECISIONS.md`](docs/DECISIONS.md) · API: [`docs/API.md`](docs/API.md) ·
Kết quả đánh giá: [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md)

## Kiến trúc

```
web (Next.js 14, :3000) ──SSE──▶ api (FastAPI, :8000)
                                   ├─ Haiku: phân loại ý định + viết lại câu hỏi
                                   ├─ BM25 (bm25s + pyvi)  ┐ RRF ─▶ bge-reranker-v2-m3 ─▶ top-k
                                   ├─ Qdrant (bge-m3)     ┘
                                   ├─ Sonnet: sinh câu trả lời có [#i] (prompt caching)
                                   └─ Postgres: documents, chunks, query_logs, feedback, eval_runs
ingest / eval: container một lần (python -m legalrag.ingest | legalrag.eval)
```

Dữ liệu: [`YuITC/Vietnamese-Legal-Documents`](https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents)
(kho lớn, mặc định lấy 20 000 văn bản) và [`Tamir39/vietnam-tax-qa`](https://huggingface.co/datasets/Tamir39/vietnam-tax-qa)
(5 luật thuế, 600 passage, 359 câu hỏi có đáp án).

## Yêu cầu hệ thống

- Docker Desktop (Compose v2). CPU 8 nhân, **16 GB RAM** (ảnh `api` nạp bge-m3 + reranker ≈ 5 GB RAM).
- Ổ đĩa trống ≈ 15 GB (ảnh Docker ~2 GB, model ~4.5 GB, dữ liệu thô ~250 MB, index).
- Khóa Anthropic API. Tùy chọn: GPU NVIDIA + NVIDIA Container Toolkit (`docker-compose.gpu.yml`).

## Chạy trong 5 bước

```bash
git clone <repo> && cd Vietnamese-Legal-QA-RAG
cp .env.example .env            # điền ANTHROPIC_API_KEY=sk-ant-...  (và ADMIN_TOKEN)
make models                     # tải bge-m3 + bge-reranker-v2-m3 vào ./data/models (~4.5 GB, một lần)
make ingest-tamir               # nạp 5 luật thuế: build + index (~3 phút CPU)
make up                         # api :8000, web :3000 (lần đầu nạp model ~1-2 phút → /health/ready = 200)
```

Mở <http://localhost:3000> và hỏi, ví dụ *“Thu nhập nào được miễn thuế thu nhập cá nhân?”*. Câu trả lời stream
theo thời gian thực; chip `[#1]` màu xanh = trích dẫn được kiểm chứng, vàng = yếu, đỏ = không tồn tại (đã loại).
Nhấn chip để mở nguồn (tên văn bản, Điều/Khoản, link văn bản gốc, nút “Xem toàn bộ Điều”).

Các trang khác: `/search` (so sánh hybrid / dense / bm25), `/admin` (thống kê, trigger ingest, phản hồi 👎).

### Kho đầy đủ (20 000 văn bản YuITC)

```bash
make ingest                     # download + build + index; CPU: nhiều giờ, GPU: ~15 phút
```

`CORPUS_MAX_DOCS` trong `.env` giới hạn số văn bản (0 = toàn bộ 261k). Ingest ghi checkpoint
`data/processed/ingest_state_*.json`; sau Ctrl+C chạy lại với `--resume`:

```bash
docker compose run --rm ingest index --strategy article --resume
```

Chạy trên GPU host (không Docker): `source scripts/local-env.sh`, tạo venv có torch CUDA (xem
[`docs/DECISIONS.md`](docs/DECISIONS.md) ADR-007) và `DEVICE=cuda python -m legalrag.ingest index`.

## Thêm văn bản của bạn

```bash
docker compose run --rm ingest add-file --path /data/raw/my_law.txt \
    --title "Luật Xyz 12/2025/QH15" --url "https://vanban.chinhphu.vn/..."
```

Tệp text thuần, giữ các dòng `Chương I ...`, `Điều 5. ...`, `1. ...`, `a) ...` để parser nhận cấu trúc.
Sau khi chạy xong, API tự nạp lại BM25 (nếu dùng `/admin/ingest`) hoặc khởi động lại `api`.

## Đánh giá (Task 4)

```bash
make eval                                   # suite default: retrieval E0+E2+E3, answer E0 vs cấu hình tốt nhất
docker compose run --rm eval retrieval --config E1 --dataset tamir_test
docker compose run --rm eval answer --config E2-hybrid_rerank --dataset tamir_test,refusal_set --yes
docker compose run --rm eval report --runs latest   # sinh docs/EVAL_REPORT.md + docs/figures/*.png
docker compose run --rm eval errors --run <run_id> --worst 20
```

Câu trả lời và điểm judge được cache trong Redis (`LLM_CACHE_ENABLED=true` cho eval) → chạy lại không tốn token.
Trước khi gọi LLM, CLI in ước lượng chi phí (theo `PRICE_INPUT_PER_M`, `PRICE_OUTPUT_PER_M`).

## Phát triển

```bash
cd backend && uv sync --all-groups        # Python 3.11, venv .venv
make infra                                 # chỉ postgres/redis/qdrant
make test                                  # unit (không cần model/dịch vụ)
make test-integration                      # cần `make infra`
make lint                                  # ruff + mypy
source scripts/local-env.sh && cd backend && uv run python -m legalrag.cli ask "Thuế suất GTGT phổ thông là bao nhiêu?"
cd frontend && npm install && npm run dev  # UI dev server, NEXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
```

Cấu trúc: `backend/legalrag/{ingest,retrieval,generation,api,eval}` (xem DESIGN §1.6), `frontend/` (Next.js).

## Troubleshooting

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `api` bị kill / OOM khi khởi động | Thiếu RAM cho bge-m3 + reranker. Tăng RAM Docker ≥ 8 GB, hoặc `RERANK_ENABLED=false`. |
| `/health/ready` trả 503 lâu | Lần đầu nạp model 1-2 phút (CPU). Kiểm tra `make logs`; 503 kèm `qdrant:false` → `docker compose up -d qdrant`. |
| Câu trả lời chậm (~25 s truy xuất) trên CPU | Reranker cross-encoder tốn ~0,75 s/cặp × 30 ứng viên (đo: p50 23,5 s). Đặt `RERANK_TOP_N=10` (~7 s), `RERANK_ENABLED=false` (~0,1 s, hybrid RRF), tắt rerank trong “Cài đặt nâng cao”, hoặc dùng GPU (`docker-compose.gpu.yml`, ~0,4 s). |
| UI báo “Không thể xác thực với dịch vụ Anthropic API” | `ANTHROPIC_API_KEY` sai/hết hạn trong `.env`; sửa rồi `make up`. |
| “Dịch vụ ngôn ngữ đang quá tải” | 429/529 từ Anthropic; hệ thống đã retry 5 lần, thử lại sau. |
| Ingest YuITC quá chậm | CPU ~1,5 s/chunk. Giảm `CORPUS_MAX_DOCS` (vd 5000) hoặc dùng GPU. |
| `make ingest` lỗi tải HF | Mạng/HF rate limit; đặt `HF_TOKEN` trong `.env` hoặc chạy lại. |
| Windows: lỗi encoding khi chạy CLI ngoài Docker | Dùng `python -X utf8` (đã đặt trong `scripts/local-env.sh`). |

## Giấy phép dữ liệu

YuITC/Vietnamese-Legal-Documents (MIT, dẫn xuất từ BKAI-Legal-Retrieval); Tamir39/vietnam-tax-qa (CC-BY-4.0).
Câu trả lời của hệ thống chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.

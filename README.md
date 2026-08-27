# Hebrew Document Extraction — Printed + Handwritten

Extract **printed Hebrew** *and* **handwritten Hebrew** from scanned documents using two
Azure AI engines together:

| Engine | Role |
| --- | --- |
| **Azure AI Content Understanding** | OCR of printed Hebrew + layout + structured field extraction, with confidence scores |
| **Azure OpenAI GPT vision** (`gpt-5.4`, `gpt-5.6-luna`, or `gpt-5.6-sol`) | Reads Hebrew **handwriting** (which classic OCR does not support) and separates printed vs handwritten text |
| **Ivrit OCR HASH v1** (optional local sidecar) | Independent CNN-BiLSTM-CTC Hebrew reading used as advisory spelling evidence during GPT visual adjudication |

The two outputs are merged side-by-side so you can compare how each engine read the document.

> **Why two engines?** Azure's deterministic OCR (Content Understanding / Document Intelligence)
> supports **printed** Hebrew but **not** Hebrew **handwriting**. A multimodal LLM (GPT vision)
> fills that gap. Content Understanding stays the reliable source for the printed form + fields;
> GPT recovers the handwritten values.

## Architecture

```mermaid
flowchart LR
    A[Scanned PDF] --> B[Render page to PNG<br/>PyMuPDF]
    A --> C[Content Understanding<br/>custom analyzer]
    B --> D[GPT-5.x vision<br/>Responses API]
    B --> H[Ivrit OCR HASH v1<br/>local sidecar]
    C --> E[Printed text + fields<br/>+ confidence]
    D & H --> F[Handwritten text + fields]
    E & F --> G[out/result.json]
```

## Prerequisites

- Python 3.9+
- Azure CLI, logged in: `az login`
- Optional: Docker Desktop or another Docker-compatible runtime for the local HASH v1 sidecar
- Access (Entra ID RBAC) to:
  - a **Content Understanding** resource (`Cognitive Services User`)
  - an **Azure OpenAI** resource with a vision model deployment (`Cognitive Services OpenAI User`)

Authentication is **Microsoft Entra ID only** — no API keys are stored in the repo.

## Quick start for team members

```bash
git clone https://github.com/roie9876/content-understanding-hebrew.git
cd content-understanding-hebrew
make setup
cp .env.example .env
# Edit .env with your team's Azure resource names, then authenticate.
az login
make check
make run
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The repository is private, so each teammate
needs GitHub access plus the Azure RBAC roles listed above. Real documents, human transcriptions,
generated outputs, and local configuration are intentionally not distributed through Git. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the team workflow and [`docs/SECURITY.md`](docs/SECURITY.md)
before sharing data or changing repository visibility.

## Web UI

The local review workspace supports document upload, live page/stage progress, page-level
token and timing metrics, and side-by-side inspection of the original page against both
Content Understanding and the final red handwriting overlay.

```bash
pip install -r requirements.txt
az login
python app.py
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Temporary uploads are kept under
`.ui_data/`. Every analysis is saved under `analysis_results/` using the original filename
and local start time, for example `document.pdf__2026-07-12_10-30-45/`. Each run contains
all page artifacts plus a `run.json` manifest and is excluded from Git because results may
contain PII.

### Optional Ivrit OCR evidence

The pipeline can query the HASH v1 service from
[roie9876/heb-handfree](https://github.com/roie9876/heb-handfree). Keep it as a sibling clone so
its Python 3.11+/PyTorch environment stays isolated from this Python 3.9 application:

```bash
git clone https://github.com/roie9876/heb-handfree.git ../heb-handfree
cd ../heb-handfree
brew install uv
make install
IVRIT_OCR_DEVICE=mps .venv/bin/python -m uvicorn \
  ivrit_ocr.api:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 16
```

Keep that command running in a separate terminal. On a machine with Docker, the containerized CPU
service is an alternative:

```bash
cd ../heb-handfree
docker compose up --build -d
curl --fail http://127.0.0.1:8000/health
```

The local clone used for this integration was inspected at commit `b63051c`. Each page checks
`GET /health` once. When ready, original region crops are sent to `POST /v1/ocr` while Core4 runs.
The raw response is saved as `region_*/ivrit_ocr.json`, including line text, uncalibrated
confidence, crop-relative boxes, and the advisory-gate decision. HASH text is shown to GPT as
`ivrit_hash_v1` spelling evidence only when all of these deterministic checks pass:

- Core4 and HASH are both single-line, nonnumeric Hebrew readings.
- The field is not an explicit date, account, reference, amount, mark, or signature; a noisy generic
  `number` detector label does not veto two visibly nonnumeric Hebrew readings.
- Both readings contain at least three Hebrew letters and have at least 60% length overlap.
- Their punctuation- and diacritic-free Hebrew character agreement is at least 40%.
- An exact Core4-HASH duplicate is omitted because it adds no spelling information.

All other HASH responses remain visible in the **Ivrit OCR** tab but are excluded from the GPT
prompt. HASH is also excluded from deterministic agreement, faithful-source guards, and standalone
fallbacks. Its uncalibrated model confidence is never sent to GPT. If the service is disabled,
absent, or times out, the normal GPT pipeline continues unchanged.

Contextual rereads may use nearby printed labels to understand a field, but they cannot replace an
exact transcription independently supported by the visual decision and an eligible HASH reading.
Core4 must either agree exactly or have supplied the measured character correlation that admitted
HASH through the advisory gate. A conflicting contextual reading is retained as an alternative,
the selected visual text is preserved, and the region is marked for review. This prevents semantic
labels such as `עיר` (`city`) from turning agreed visible strokes into a more familiar place name.

For compact blue handwriting, InkSight also detects a specific segmentation failure: HASH returns
multiple vertically overlapping lines while Core4 sees one nonnumeric Hebrew line. Only in that
case, InkSight retries HASH on a tight blue-only image that removes black form print. The blue-only
result replaces the original HASH result only when it collapses to one line and retains at least
55% Hebrew character agreement and 80% length overlap with the original HASH reading. This recovers
line samples that were trained/evaluated as tight crops without globally applying a transform that
regresses other fields. The Ivrit OCR tab records `original` or `blue_only` as the selected input.

Configure the sidecar in `.env`:

```dotenv
IVRIT_OCR_ENABLED=1
IVRIT_OCR_URL=http://127.0.0.1:8000
IVRIT_OCR_TIMEOUT=60
IVRIT_OCR_WORKERS=1
IVRIT_OCR_MIN_AGREEMENT=0.40
IVRIT_OCR_MIN_PREPROCESS_AGREEMENT=0.55
```

HASH v1 is an assisted-transcription model, not an authority. Its model card reports 45.95% CER
and 100% WER on the confirmed natural ruled-form evaluation, and its confidence is not calibrated.
The **Ivrit OCR** tab therefore exposes its evidence independently; final text still requires GPT
to verify the visible strokes. Preserve the upstream `NOTICE.md` when redistributing its checkpoint
or container. Stop the native Uvicorn process with Ctrl+C, or use `docker compose down` for Docker.

### Experimental Kraken PP-OCRv6 ablation

The **Experimental standard + Kraken medium advisory** OCR strategy runs the normal Core4 + visual
reconciliation pipeline and adds a local PP-OCRv6-medium line reading as weak evidence. Kraken never
replaces GPT directly. Empty, numeric, multiline, duplicate, and weakly correlated readings are
saved for inspection but excluded from visual adjudication. The standard strategy keeps Kraken off.

Install the ignored local environment and Apache-2.0 checkpoint once:

```bash
uv venv .models/kraken-venv --python 3.12
.models/kraken-venv/bin/python -m pip install 'kraken>=7.1.0'
mkdir -p .models/pp-ocrv6-medium
curl -L --fail 'https://zenodo.org/records/21788410/files/medium.safetensors?download=1' \
  -o .models/pp-ocrv6-medium/medium.safetensors
test "$(md5 -q .models/pp-ocrv6-medium/medium.safetensors)" = \
  'e3411a453ce3b9e9efae3f8631d85762'
```

For a controlled UI comparison, analyze the same document pages twice with the same vision model:

1. Select **Standard core4 + visual reconciliation** for the control run.
2. Select **Experimental standard + Kraken medium advisory** for the treatment run.
3. Compare each run's `handwriting_result.json` with the same local human golden using
  `python3 compare_golden.py --result PATH --golden-stem STEM`.

Each treatment region records raw text and gate diagnostics in `kraken_ocr.json`; eligible Kraken
text also appears in `visual_reconcile_input.json`. CPU inference is intentional because Kraken 7.1
mixes CPU and MPS tensors on this model. This strategy is an ablation tool, not a quality default.

The upload screen lists completed local analyses. Selecting one reopens the full source,
detection, candidates, decision, final overlay, and logs without calling Azure again. Saved
runs are restored from their manifests whenever the server starts.

To reproduce the isolated multi-view experiment from the UI, upload a blue-handwritten lined
image, select either **GPT-5.4** (medium reasoning) or **GPT-5.6 Sol · low reasoning**, and set
**OCR strategy** to
**Experimental isolated multi-view · lined handwriting** before selecting **Analyze selected**.
The strategy isolates lined rows, uses stable word groups only when geometry supports them, sends
three faithful views of each unit through two candidate-blind reads, and aligns the results in
code. Regions without valid colored lined geometry automatically use standard OCR. Experimental
jobs use a longer timeout, run isolated calls sequentially, and can take substantially longer.
GPT-5.4 uses a 120-second per-endpoint timeout; Sol-low uses 240 seconds. The strategy runs at
most two independent draws concurrently. GPT-5.4 receives a larger output budget so reasoning
does not repeatedly exhaust a Sol-sized cap and trigger endpoint failover. Stable word groups run
before expensive whole-line audits. Other model profiles use the standard strategy.

For the standard strategies, GPT-5.4 handwriting OCR uses low reasoning and a 4,000-token budget.
Detection preserves the quality profile: medium reasoning, a 6,000-token budget, a 90-second
per-endpoint timeout, and failover across the configured endpoint pool. If the full-page scan
fails, five overlapping coverage segments run sequentially and report visible progress in the UI.

The UI hides Content Understanding's internal Markdown comments and figure descriptions while
preserving the raw artifact on disk. Every upload performs fresh Azure analysis; OCR responses,
Content Understanding responses, and completed document results are never replayed.

For PDFs, upload is followed by a page-selection workspace before any Azure analysis
starts. Users can inspect page previews, click individual page thumbnails, select all
or clear the selection, or enter lists/ranges such as `1,3-5`. Only selected pages are
scheduled and billed, while their original document page numbers are preserved in the
results and combined Markdown.
Multi-page uploads process two pages concurrently by default. The existing eight-worker
OCR budget is divided between them, so handwriting still uses the same medium reasoning,
full image resolution, enhancement views, and self-consistency voting. Tune with
`PAGE_CONCURRENCY` and `MAX_WORKERS`; set `PAGE_CONCURRENCY=1` to restore sequential page
processing. Each job also writes a persistent `.ui_data/outputs/<job-id>/pipeline.log`.

The page workspace exposes the complete analysis trail for administrators:

- **Content Understanding** appears as soon as its background call completes.
- **Detection** lists every handwriting region with its page coordinates and numbered overlay.
- **Candidates** updates after each vision draw and groups alternate readings by region.
- **Ivrit OCR** shows HASH v1 availability, text, line confidence, crop-relative boxes, and whether
  the deterministic gate sent or excluded that reading from GPT evidence.
- **Decision** records the selected transcription, voting method, alternatives, word confidence,
  and visual reasons.
- **Final** shows the red handwriting overlay; **Logs** shows page-scoped pipeline events.

While a page is active, the header reports child-process liveness, completed/total draws,
failed or retrying draws, and the age of the latest pipeline event. Finished artifacts are also
used to reconcile stale in-memory status so a completed page cannot remain displayed as running.

## Setup

```bash
pip install -r requirements.txt
az login
# optional: cp .env.example .env   and edit
```

## Usage

```bash
python extract_hebrew.py "sample/אישור מסירה -דירה 8.pdf"
```

Run Content Understanding only (skip GPT):

```bash
SKIP_GPT=1 python extract_hebrew.py "sample/אישור מסירה -דירה 8.pdf"
```

Higher-resolution render (better for faint handwriting):

```bash
RENDER_DPI=300 python extract_hebrew.py "sample/אישור מסירה -דירה 8.pdf"
```

## Output

Everything is written to the **`out/`** directory (git-ignored):

| File | Description |
| --- | --- |
| `out/result.json` | The merged, structured result (see schema below) |
| `out/page_N.png` | Each PDF page rendered to an image (also sent to GPT) |

`out/result.json` structure:

```jsonc
{
  "source_file": "sample/אישור מסירה -דירה 8.pdf",
  "content_understanding": {
    "markdown": "…full printed-Hebrew OCR of the form…",
    "fields": {
      "DocumentDate":    { "value": "2026-01-01", "confidence": 0.97 },
      "BuildingNumber":  { "value": "4",          "confidence": 0.97 },
      "ApartmentNumber": { "value": "8",          "confidence": 0.98 },
      "RecipientName":   { "value": null,         "confidence": 0.78 },
      "HandwrittenNotes":{ "value": null,         "confidence": 0.78 }
    }
  },
  "gpt_vision": [
    {
      "page": 1,
      "printed_text": "…",
      "handwritten_text": "…קיבלתי מזגן…",
      "fields": {
        "document_date": "…",
        "apartment_number": "8",
        "building_number": "4",
        "handwritten_notes": "…",
        "is_signed": true,
        "has_stamp": false
      },
      "uncertain": "words the model was unsure about"
    }
  ]
}
```

A short summary is also printed to the console after each run.

## Repository structure

```
.
├── extract_hebrew.py        # main pipeline (CU + GPT vision + merge)
├── handwriting_pipeline.py  # focused multi-pass handwriting OCR helper
├── compare_golden.py        # regression check: current output vs golden/
├── EXPERIMENT_LOG.md        # dated results, regressions, costs, and decisions
├── exp_determinism.py       # stability probe: single-draw vs self-consistency vote
├── azure_ocr_benchmark.py   # Azure Document Intelligence vs GPT-voting benchmark
├── requirements.txt         # Python dependencies
├── package.json             # optional GitHub Copilot SDK integration
├── Makefile                 # setup, test, check, and run commands
├── CONTRIBUTING.md          # team workflow and data-handling rules
├── .env.example             # configuration template
├── .gitignore
├── tests/                   # offline unit tests
├── sample/                  # example input documents
├── golden/                  # approved results per sample (regression baseline)
├── out/                     # extract_hebrew.py results (git-ignored)
├── output_handwriting/      # handwriting_pipeline.py results (git-ignored)
└── results/                 # <name>_results.jpg per document (git-ignored)
```

Real source documents and human golden transcriptions live locally under `sample/` and
`golden/`. Both directories are ignored except for their README files because they can contain
names, account numbers, IDs, handwriting, and other personal data. Use synthetic or fully
anonymized fixtures for anything committed to Git.

Every model, prompt, preprocessing, segmentation, voting, or adjudication experiment must add a
dated entry to `EXPERIMENT_LOG.md`, including negative and inconclusive results.

## Handwriting pipeline (`handwriting_pipeline.py`)

When handwriting is hard to read, this pipeline boosts accuracy by **not** relying on a
single OCR pass. Per handwriting region it:

1. **Detects** handwriting regions automatically (or takes manual crop boxes).
2. Creates and saves seven **enhanced image versions** plus **overlapping chunks** for review.
3. Sends the four benchmarked core views (original, upscaled contrast, denoised, and
  line-removed) together in one **GPT-5.x vision** call per region. Optional transforms
  are never retried; a stranded region gets bounded original/denoised rescue calls.
4. **Visually adjudicates** the final reading: it re-shows the best images + all candidate
   readings to the model and lets it decide from the strokes (not a blind majority vote),
   returning per-word **confidence + a visual reason**. Falls back to text-only
   reconciliation, then to the best per-image reading.

It is **domain-agnostic** — no built-in vocabulary, no assumption about document type. You
can optionally pass a short **printed-text context** (from the Content Understanding output)
as a *soft* hint with `--printed-context-file` + `--context-mode nearby`; it never overrides
the visible strokes.

### How to test it

**The tool is smart about input — `--input` alone does the right thing.** No `--page`,
`--auto-detect`, or `--merge-md` needed:

```bash
az login   # once

# single page or image — auto-detects handwriting + builds merged.md
python handwriting_pipeline.py --input "sample/אישור מסירה -דירה 8.pdf" \
  --output ./output_handwriting/test

# a multi-page PDF — ALL pages are processed automatically
python handwriting_pipeline.py --input report.pdf --output ./output_handwriting/report
#   -> output_handwriting/report/page_0001/…, page_0002/…, and a combined document.md
```

Restrict or tune only when you want to: `--pages 1-3,5` (subset), `--page 2` (one page),
`--crop "x,y,w,h"` (manual region, skips auto-detect), `--no-merge` (skip merged.md),
`--workers N`, `OCR_REASONING=low` (faster). For a big PDF, note it runs per page
(~N× the single-page time).

**Inspect the outputs** (inside `--output`, git-ignored). For a multi-page PDF each page is
in its own `page_NNNN/` folder plus a top-level `document.md` concatenating them:

| File | What it is |
| --- | --- |
| `page_1_regions.png` | the page with numbered red detection boxes |
| `region_NN_*/result.json` | the region's final reading + per-word confidence/reason/notes |
| `region_NN_*/per_image_ocr.json` | what each enhanced image/chunk read |
| `handwriting_result.json` | all regions combined (+ `usage`, `elapsed`) |
| `merged.html` / `merged.jpg` | the page with extracted handwriting overlaid in **red** |
| `merged.md` | full page: CU printed text + handwriting in red, de-duplicated |
| `document.md` | (multi-page) all pages' `merged.md` concatenated |


**Other ways to run:**

```bash
# Manual crop(s) instead of auto-detect (repeat --crop per region):
python handwriting_pipeline.py --input "sample/doc.pdf" \
  --crop "180,1660,2050,260" --ocr-mode both --output ./output_handwriting/test

# No coordinates yet? Run with no crop to render the page, then read the pixel
# coords off page_1_render.png and re-run with --crop:
python handwriting_pipeline.py --input "sample/doc.pdf" --output ./output_handwriting/test

# Also build the merged Markdown (runs Content Understanding for the printed layer):
python handwriting_pipeline.py --input "sample/doc.pdf" --auto-detect --merge-md \
  --output ./output_handwriting/test

# Debugging: one call at a time so per-image outputs are easy to compare:
python handwriting_pipeline.py --input "sample/doc.pdf" --auto-detect --workers 1 \
  --output ./output_handwriting/test
```

**Batch — run all sample files, save `results/<name>_results.jpg`:**

```bash
mkdir -p results
for f in sample/*.pdf sample/*.png; do
  stem=$(basename "$f"); stem="${stem%.*}"
  python3 handwriting_pipeline.py --input "$f" --auto-detect --ocr-mode full \
    --workers 8 --output "output_handwriting/$stem"
  cp "output_handwriting/$stem/merged.jpg" "results/${stem}_results.jpg"
done
```

> **Tip:** run files **sequentially** (as above) or keep `--workers` modest so concurrent
> runs don't compete for the shared GPT quota — contention can cause 429s and empty regions.

### CLI options (`--help` for the full list)

- *(defaults)* no crop → **auto-detect** handwriting; a PDF → **all pages**; **merged.md on**
- `--page N` / `--pages "1-3,5"` — process one page / a subset (default: all pages)
- `--crop "x,y,w,h"` — manual region in pixels (repeat for multiple regions; skips auto-detect)
- `--no-merge` — skip `merged.md`
- `--ocr-mode full|chunks|both` — which images to OCR (default `full`)
- `--workers N` — parallel GPT calls in flight (`1` for easy debugging)
- `--printed-context / --printed-context-file` + `--context-mode none|nearby|summary` —
  optional short printed-text **soft** hint (default `none`)
- `--italic` — also read italic/cursive **printed** Hebrew here (opt-in only; normally
  italic is printed text and belongs to the Content Understanding pipeline)
- `--dpi`, `--max-regions`

The handwriting engine is a Vision LLM — Tesseract is intentionally not used (weak for
Hebrew handwriting).

## Quality authority and optional voting

The default quality path reads four core views in one multi-image call, then performs final
**visual adjudication** with the image evidence attached again. Candidates are hints, while
visible strokes remain the authority. Deterministic guards preserve numeric cell order, prevent
the adjudicator from erasing faithful non-empty text, and classify high-specificity signature
flourishes and X/checkbox marks explicitly. All seven transforms are still saved for inspection.

Set `OCR_MULTI_IMAGE=0` to run the four core views as independent calls. That mode is slower but
is retained as a measured quality-first A/B fallback. No OCR, document, or golden-result cache is
used in either mode.

Blind contextual rereads that change a non-empty visual result must meet
`CONTEXTUAL_RECHECK_MIN_CONFIDENCE` (default `0.60`). Disagreements are retained in the evidence
and marked for review; the threshold can be raised for more conservative automatic acceptance.
Repeated identifier fields also provide a document-local consistency check: equal-length numeric
values that differ by exactly one digit are aligned only when the stronger field has at least
`0.90` confidence and a `0.10` confidence margin. Dates and free text are never copied this way.

Two later quality experiments are retained but disabled by default:

- `OCR_SPATIAL_NUMERIC_CELLS=1` asks contextual adjudication for cell values and normalized
  x-coordinates, then assembles valid numeric cells deterministically from left to right.
- `OCR_NEARBY_CU_CONTEXT=1` adds a local reading-order window from Content Understanding as a
  soft printed-text hint. This requires merged/CU output and remains experimental because CU
  Markdown does not include reliable spatial coordinates.

GPT-5.4 can use a two-endpoint pool configured locally in `.env`.

The UI also offers high-cost experimental
`Claude Opus 4.8 · GitHub Copilot SDK · Core4`. It uses the standard four-view Core4 OCR call and
visual reconciliation through the
official `@github/copilot-sdk` Node package and image blob attachments, not an Azure/model API
endpoint. Set `COPILOT_GITHUB_USER` to a login whose `gh` authentication has Opus 4.8 enabled,
then run `npm install`. The app obtains that account's token with `gh auth token --user`, passes it
only to the SDK child process, and disables Keychain probing. Tools, skills, memory, config
discovery, and persistent Copilot sessions are disabled. Copilot calls consume the selected
account's premium usage allowance; Opus 4.8 is marked high cost in Copilot.

`Gemini 3.5 Flash · GitHub Copilot SDK` uses the same account-specific, Keychain-free bridge with
the exact enabled vision model `gemini-3.5-flash`. It uses low reasoning and a shorter timeout as
a speed-oriented experimental profile. GPT-5.4 remains the default until golden evaluation proves
that another profile improves quality.

Calls start on each endpoint in round-robin order. Connection failures, timeouts, 408/409/429,
and transient 5xx responses immediately fail over to the other endpoint before normal backoff.
Successful calls per endpoint are recorded under `usage.targets` in `handwriting_result.json`.
Set `GPT_TARGETS` to a JSON array of `{name, endpoint, deployment}` objects to override the pool.

Self-consistency voting remains available as an opt-in experiment with `VOTE=1`. It draws each
image `VOTE_SEEDS` times and aggregates candidates deterministically. Structured values use
format-constrained voting; free text uses a medoid and per-token agreement. This can improve
stability, but it is not the default because text-only voting cannot recover strokes omitted by
a crop or resolve visual ambiguity as well as the final image-aware adjudicator.

**Why not a deterministic OCR engine?** `azure_ocr_benchmark.py` benchmarks Azure Document
Intelligence (`prebuilt-read`) on the same crops: it is 100% deterministic but inaccurate on
Hebrew handwriting (reads the printed body, garbles the cursive, sim ~0.1–0.4). A
deterministic engine trades too much accuracy, so GPT voting wins.

### Speed vs quality

Measured on the 21-region human-verified `חיוב חשבון1` sample with GPT-5.4 medium reasoning:

| Pipeline | Exact text | Mean similarity | Signature | Time | GPT calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Original seven-view baseline | 5/20 | 0.683 | 0/1 | 26:20 | 167 | 475,850 |
| Quality-fixed seven-view checkpoint | 5/20 | 0.744 | 1/1 | 18:53 | 168 | 496,472 |
| Four independent core views | **7/20** | 0.746 | 1/1 | 12:17 | 105 | 341,970 |
| Four-view multi-image default | 6/20 | **0.767** | 1/1 | **09:10** | **44** | **177,818** |

The multi-image default cuts wall time by about 65%, calls by 74%, and tokens by 63% versus
the original baseline while improving both exact accuracy and mean similarity. The independent
mode produced one more exact field in this run; use it when that quality/cost tradeoff is preferred.

Adaptive highlighted-context experiments on the same 21-region sample:

| Experiment | Exact text | Mean similarity | Signature | Time | GPT calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Fresh baseline | 6/20 | 0.749 | 1/1 | 16:16 | 45 | 188,224 |
| Highlighted context v1 | 7/20 | 0.699 | 1/1 | 13:12 | 55 | 211,493 |
| **Conservative highlighted context v2** | **8/20** | **0.798** | 1/1 | 21:47 | 61 | 255,899 |
| Spatial numeric cells | 7/20 | 0.814 | 1/1 | 12:15 | 61 | 252,607 |
| Nearby CU reading-order context | 7/20 | 0.750 | 1/1 | 13:31 | 59 | 253,120 |

Conservative highlighted context v2 is the accepted checkpoint because it improved exact accuracy
and mean similarity together. Spatial cells improved mean similarity but lost one exact field, so
they remain opt-in. Nearby CU context regressed because Markdown reading order is not a reliable
page-coordinate map; a future version should consume CU polygons before this can be enabled safely.

Measured on `אישור` (8 regions, 16 workers):

| Setting | Time | Cursive quality |
|---|---|---|
| **medium reasoning** (default) | ~5–6 min | best — no empty/garbled fields |
| `OCR_REASONING=low` | ~2 min | degrades: some cursive fields go empty/garbled |
| `OCR_REASONING=low VOTE_SEEDS=5` | ~4 min | partially recovers, still below medium |

The bottleneck is **per-call reasoning latency**, not parallelism (crops already run in
parallel up to `--workers`). Lowering reasoning is the only large speedup, but it costs
cursive accuracy — so **medium is the default**. `GPT_READ_TIMEOUT` (default 90 s) caps a
stuck call so it retries fast, taming worst-case tail latency. Speed/quality knobs:
`OCR_REASONING`, `VOTE_SEEDS`, `--workers`, `GPT_READ_TIMEOUT`.

### Golden dataset & regression checks

`golden/` holds approved reference results. It is used only for regression comparison; every run
performs fresh Azure analysis. The report includes strict normalized exact-text accuracy over all
human text regions and explicit non-text classification accuracy in addition to the legacy
structural verdict:

```bash
python3 compare_golden.py         # regression check: current output vs golden
python3 compare_golden.py --run   # re-run all registered samples, then compare
```

The comparator fails only on **structural** regressions (missing/empty regions, count change)
or a **stable** field (dates, numbers, IDs, emails) changing; volatile cursive/`[?]` fields
are reported but never failed, since the model rewords them run-to-run.

For deterministic quality-rule development, `quality_replay.py` reapplies current detector
deduplication, HASH selection, numeric/date consensus, signature geometry, and output filtering to
saved OCR evidence. It makes no model calls and loads golden answers only after all transformations:

```bash
python3 quality_replay.py \
  --run 'חיוב חשבון1=analysis_results/חיוב חשבון1.pdf__2026-07-15_09-58-19' \
  --run 'דוח_יוסי=analysis_results/דוח_יוסי.pdf__2026-07-15_11-27-12'
```

Repeat `--run 'GOLDEN_STEM=SAVED_RUN_DIR'` for any set of documents; add `--json` for
machine-readable output. The final eight-document replay improved strict exact text from
20/70 to 31/70, mean similarity from 0.646 to 0.704, and non-text classification from 5/7 to
7/7. On the two newly evaluated documents, current deterministic rules produce 3/3 exact fields
for `חוזר מכר` and retain the one unresolved `פרוטוקול` date without inventing a correction.

CU text used by deterministic numeric reconciliation is kept separate from model-facing nearby
context. A unique one-edit CU token may repair a typed numeric field after visual adjudication;
CU is not inserted into the model prompt unless `OCR_NEARBY_CU_CONTEXT=1`. A second
high-reasoning reread is even narrower: it runs only when non-signature text remains explicitly
`[?]` with no resolved characters, and accepts only Hebrew `handwritten_text` at confidence
`>=0.70`. Empty `no_handwriting` detections are omitted from exported/merged regions only when all
blind vision reads are also empty; their diagnostics remain in `rejected_regions`.

When Core4 returns only digits but independent HASH output contains substantial Hebrew in a
name/text slot, the pipeline performs two independent candidate-blind script audits before the
normal contextual reread. Each audit uses one model endpoint and receives the tight crop, blue-only
support, larger context, and highlighted page without either OCR candidate. A Hebrew name is emitted
only when both audits exactly agree at sufficient confidence. Hebrew disagreement, mixed-script
disagreement, a partial timeout with one Hebrew result, or two endpoint timeouts emit `[?]` with
`needs_review` instead of preserving false numeric text or inventing a plausible name.
The audit is intentionally narrow: strong name-slot labels, or a generic colored-handwriting label
with at least three overlapping HASH fragments. Dates, accounts, amounts, IDs, and normal numeric
fields are excluded.

For crop/detection changes, run the detector without OCR or reconciliation. It matches current
boxes against human text-region geometry, supports tight crops inside older broad/overlapping
golden boxes, and fails on missed regions, excessive extras, or large center drift:

```bash
python3 detection_regression.py
python3 detection_regression.py --stem "דוח_כחלון" --stem "דוח_אלעד"
```

The command writes an ignored report plus detector overlays under `analysis_results/`. A local
`golden/<document>/detection_regions.json` may define detection-only targets without storing or
inventing OCR text. Golden geometry remains evaluation-only and is never read by the runtime
pipeline.

### Full-agent model benchmark

`agent_benchmark.py` runs a GitHub Copilot SDK agent end to end against the rendered page. The
agent performs detection, transcription, classification, and merged Markdown generation in one
tool-using session. The orchestrator enforces a fixed profile matrix and scores each immutable
result only after inference, so human golden text is never included in the agent prompt or tools.

```bash
# Default: Sol low/medium/high plus GPT-5.4 medium
python3 -u agent_benchmark.py --input "sample/חיוב חשבון1.pdf" --page 1

# Any explicit subset of the six available profiles
python3 -u agent_benchmark.py --input "sample/חיוב חשבון1.pdf" \
  --profiles sol-low,sol-medium,sol-high,gpt-5.4-low,gpt-5.4-medium,gpt-5.4-high
```

Each run is saved under ignored `agent_benchmarks/<document>__<timestamp>/`. Every profile keeps
its request metadata, agent event stream, raw response, normalized `handwriting_result.json`,
merged Markdown, execution status, and golden score. The run root contains `summary.json` and a
ranked `report.md`. Accuracy ranks before similarity and elapsed time.

For customer environments where Gemini is unavailable, `sealed_hybrid_benchmark.py` runs an
OpenAI-only ensemble: fresh GPT-5.4 and GPT-5.6 Sol-low staged pipelines produce independent
candidates, then GPT-5.4 high reasoning adjudicates enlarged source crops. Candidate identities
are hidden from the adjudicator as `primary` and `secondary`. The inference process does not load
`compare_golden`; the evaluator is imported only after both candidate processes and the
adjudicator have exited.

```bash
python3 -u sealed_hybrid_benchmark.py \
  --input "sample/חיוב חשבון1.pdf" --page 1 --reasoning high
```

The generated `sealed_request.json` contains only neutral region identifiers, model candidates,
crop paths, and inference configuration. A leakage guard rejects evaluation-related fields before
the adjudicator starts. Anthropic is not part of this path because the measured Opus 4.8 baseline
did not outperform GPT-5.4 on the human-golden document.

The strongest repeatable customer-compatible configuration is three fresh GPT-5.6 Sol-low staged
runs followed by deterministic spatial majority consensus. It makes no model call during reduction
and never imports the golden evaluator:

```bash
python3 -u sol_consensus.py --input "sample/חיוב חשבון1.pdf" --page 1 --runs 3
```

On `חיוב חשבון1`, the three independent runs scored 11/20, 8/20, and 9/20 exact; consensus scored
10/20 exact, 84.7% similarity, and 100% non-text classification. Sol-low is therefore the normal UI
default. GitHub-only Gemini and Opus profiles remain available in backend code for controlled
experiments but are intentionally hidden from the customer-facing picker.

## Notes & limitations

- **Handwriting is inherently ambiguous.** GPT may read an unclear date/name differently across
  fresh runs. Treat uncertain handwriting as *"needs review"*, especially where candidates
  disagree or Content Understanding returns `null`.
- **Italic/cursive printed Hebrew is *printed*, not handwriting** — it belongs to the
  Content Understanding pipeline. The handwriting detector can occasionally over-detect
  italic script as handwriting; use `--italic` only as an opt-in when you want GPT to
  re-read italic (Content Understanding reads italic Hebrew less accurately).
- **Sensitive data:** the sample documents and generated results can contain personal
  information (names, ID numbers). `out/`, `output_handwriting/`, `results/`, and
  `agent_benchmarks/` are
  git-ignored; review the `sample/` files before pushing this repo to any remote.
- Content Understanding requires model deployments (`gpt-4.1` / `gpt-4.1-mini` /
  `text-embedding-3-large`) configured as defaults on the resource.

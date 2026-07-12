# Hebrew Document Extraction — Printed + Handwritten

Extract **printed Hebrew** *and* **handwritten Hebrew** from scanned documents using two
Azure AI engines together:

| Engine | Role |
| --- | --- |
| **Azure AI Content Understanding** | OCR of printed Hebrew + layout + structured field extraction, with confidence scores |
| **Azure OpenAI GPT vision** (`gpt-5.4`, `gpt-5.6-luna`, or `gpt-5.6-sol`) | Reads Hebrew **handwriting** (which classic OCR does not support) and separates printed vs handwritten text |

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
    C --> E[Printed text + fields<br/>+ confidence]
    D --> F[Handwritten text + fields]
    E & F --> G[out/result.json]
```

## Prerequisites

- Python 3.9+
- Azure CLI, logged in: `az login`
- Access (Entra ID RBAC) to:
  - a **Content Understanding** resource (`Cognitive Services User`)
  - an **Azure OpenAI** resource with a vision model deployment (`Cognitive Services OpenAI User`)

Authentication is **Microsoft Entra ID only** — no API keys are stored in the repo.

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

The upload screen lists completed local analyses. Selecting one reopens the full source,
detection, candidates, decision, final overlay, and logs without calling Azure again. Saved
runs are restored from their manifests whenever the server starts.

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
├── exp_determinism.py       # stability probe: single-draw vs self-consistency vote
├── azure_ocr_benchmark.py   # Azure Document Intelligence vs GPT-voting benchmark
├── requirements.txt         # Python dependencies
├── .env.example             # configuration template
├── .gitignore
├── sample/                  # example input documents
├── golden/                  # approved results per sample (regression baseline)
├── out/                     # extract_hebrew.py results (git-ignored)
├── output_handwriting/      # handwriting_pipeline.py results (git-ignored)
└── results/                 # <name>_results.jpg per document (git-ignored)
```

## Handwriting pipeline (`handwriting_pipeline.py`)

When handwriting is hard to read, this pipeline boosts accuracy by **not** relying on a
single OCR pass. Per handwriting region it:

1. **Detects** handwriting regions automatically (or takes manual crop boxes).
2. Creates several **enhanced image versions** (upscale+contrast, adaptive threshold,
   denoise, thickened strokes, horizontal-line removal) and **overlapping chunks**.
3. Sends every image to the selected **GPT-5.x vision model** to transcribe *only* the handwriting — all calls
   run in parallel, with automatic retries so a timeout burst never drops a region.
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

The default quality path reads all seven enhanced images once, then performs final **visual
adjudication** with the image evidence attached again. This matches the architecture used to
produce the approved golden baseline: candidates are hints, while visible strokes remain the
authority. If visual adjudication fails, the pipeline falls back to text reconciliation and then
the best non-empty candidate. The seventh view preserves neighborhoods around thick stroke cores,
which makes handwriting easier to distinguish from thin printed form text without removing the
original crop from the evidence set.

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
performs fresh Azure analysis:

```bash
python3 compare_golden.py         # regression check: current output vs golden
python3 compare_golden.py --run   # re-run the 3 samples, then compare
```

The comparator fails only on **structural** regressions (missing/empty regions, count change)
or a **stable** field (dates, numbers, IDs, emails) changing; volatile cursive/`[?]` fields
are reported but never failed, since the model rewords them run-to-run.

## Notes & limitations

- **Handwriting is inherently ambiguous.** GPT may read an unclear date/name differently across
  fresh runs. Treat uncertain handwriting as *"needs review"*, especially where candidates
  disagree or Content Understanding returns `null`.
- **Italic/cursive printed Hebrew is *printed*, not handwriting** — it belongs to the
  Content Understanding pipeline. The handwriting detector can occasionally over-detect
  italic script as handwriting; use `--italic` only as an opt-in when you want GPT to
  re-read italic (Content Understanding reads italic Hebrew less accurately).
- **Sensitive data:** the sample documents and generated results can contain personal
  information (names, ID numbers). `out/`, `output_handwriting/`, and `results/` are
  git-ignored; review the `sample/` files before pushing this repo to any remote.
- Content Understanding requires model deployments (`gpt-4.1` / `gpt-4.1-mini` /
  `text-embedding-3-large`) configured as defaults on the resource.

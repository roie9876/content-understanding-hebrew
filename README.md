# Hebrew Document Extraction — Printed + Handwritten

Extract **printed Hebrew** *and* **handwritten Hebrew** from scanned documents using two
Azure AI engines together:

| Engine | Role |
| --- | --- |
| **Azure AI Content Understanding** | OCR of printed Hebrew + layout + structured field extraction, with confidence scores |
| **Azure OpenAI GPT vision** (`gpt-5.4`) | Reads Hebrew **handwriting** (which classic OCR does not support) and separates printed vs handwritten text |

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
    B --> D[GPT-5.4 vision<br/>Responses API]
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
├── seed_golden_cache.py     # seed the result-cache from golden/ (instant reproduction)
├── exp_determinism.py       # stability probe: single-draw vs self-consistency vote
├── azure_ocr_benchmark.py   # Azure Document Intelligence vs GPT-voting benchmark
├── requirements.txt         # Python dependencies
├── .env.example             # configuration template
├── .gitignore
├── sample/                  # example input documents
├── golden/                  # approved results per sample (regression baseline)
├── out/                     # extract_hebrew.py results (git-ignored)
├── output_handwriting/      # handwriting_pipeline.py results (git-ignored)
├── results/                 # <name>_results.jpg per document (git-ignored)
└── .cache/                  # response + result cache (git-ignored)
```

## Handwriting pipeline (`handwriting_pipeline.py`)

When handwriting is hard to read, this pipeline boosts accuracy by **not** relying on a
single OCR pass. Per handwriting region it:

1. **Detects** handwriting regions automatically (or takes manual crop boxes).
2. Creates several **enhanced image versions** (upscale+contrast, adaptive threshold,
   denoise, thickened strokes, horizontal-line removal) and **overlapping chunks**.
3. Sends every image to **GPT-5.4 vision** to transcribe *only* the handwriting — all calls
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

**Step 1 — auto-detect and transcribe everything (recommended):**

```bash
az login   # once
python handwriting_pipeline.py \
  --input "sample/אישור מסירה -דירה 8.pdf" \
  --auto-detect --ocr-mode full --workers 8 \
  --output ./output_handwriting/test
```

**Step 2 — inspect the outputs** (inside `--output`, git-ignored):

| File | What it is |
| --- | --- |
| `page_1_regions.png` | the page with numbered red detection boxes |
| `region_NN_*/result.json` | the region's final reading + per-word confidence/reason/notes |
| `region_NN_*/per_image_ocr.json` | what each enhanced image/chunk read |
| `region_NN_*/visual_reconcile_input.json` | images + candidates sent to the visual step |
| `region_NN_*/01_…`–`06_…`, `chunk_*.png` | the enhanced crops actually sent to GPT |
| `handwriting_result.json` | all regions combined |
| `merged.html` / `merged.jpg` | the page with extracted handwriting overlaid in **red** |
| `merged.md` | (with `--merge-md`) full doc: CU printed text + handwriting in red, de-duplicated |

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

- `--auto-detect` — locate all handwriting regions automatically
- `--crop "x,y,w,h"` — manual region in pixels (repeat for multiple regions)
- `--ocr-mode full|chunks|both` — which images to OCR (default `both`)
- `--workers N` — parallel GPT calls in flight (`1` for easy debugging)
- `--merge-md` — also build `merged.md` (CU printed text + handwriting, de-duplicated)
- `--no-cache` — disable the model-response cache (always call the model)
- `--refresh-cache` — ignore the cache, recompute, and overwrite it
- `--printed-context / --printed-context-file` + `--context-mode none|nearby|summary` —
  optional short printed-text **soft** hint (default `none`)
- `--italic` — also read italic/cursive **printed** Hebrew here (opt-in only; normally
  italic is printed text and belongs to the Content Understanding pipeline)
- `--dpi`, `--page`, `--max-regions`

The handwriting engine is a Vision LLM — Tesseract is intentionally not used (weak for
Hebrew handwriting).

## Consistency: self-consistency voting (deterministic where the input allows)

The Vision LLM is **non-deterministic** on ambiguous handwriting. Rather than replay a
cached guess, the pipeline computes a **stable, honest** result on every run via
**self-consistency voting**:

1. Each region is drawn **N times** (3 image versions × `VOTE_SEEDS`=3 fixed seeds = 9
   genuine OCR draws).
2. The draws are aggregated **deterministically in code** (no model call):
   - **Structured fields** (date / phone / id / email / number) → format-constrained
     majority vote. These are **truly deterministic + correct** (measured: date & numbers
     1 distinct across fresh sessions, similarity 1.00).
   - **Free text** → medoid + per-token majority: agreed tokens **lock**, the rest are
     kept as `[?]` with alternatives.

Measured on the free-text-heavy `אישור` sample (cache off, 3 fresh sessions): single draws
were 9–12 of 12 distinct (chaos); the **vote lifts cross-session similarity to 0.5–0.8** and
locks the readable skeleton (`3 מפתחות … מים`), flagging only the genuinely-ambiguous words
`[?]`. That is the honest ceiling — genuinely-ambiguous cursive can't be made byte-identical
across fresh draws, but it is never a different fabrication each run. `VOTE=0` restores the
old (non-deterministic) reconcile path; `VOTE_SEEDS` sets the draws per version.

**Why not a deterministic OCR engine?** `azure_ocr_benchmark.py` benchmarks Azure Document
Intelligence (`prebuilt-read`) on the same crops: it is 100% deterministic but inaccurate on
Hebrew handwriting (reads the printed body, garbles the cursive, sim ~0.1–0.4). A
deterministic engine trades too much accuracy, so GPT voting wins.

### Reproducibility & speed (caching)

- **Response cache** (`.cache/gpt`, `.cache/cu`) — each GPT/CU call is memoized on disk,
  keyed on its content **and seed**, so the N voting draws are distinct genuine calls and a
  **re-run of the same document reuses them → byte-identical result, 0 new tokens**.
- **Result cache** (`.cache/docs`, opt-in with `DOC_CACHE=1`) — stores the whole output dir
  per input for instant re-serve. **Off by default** (we want genuine computation).
- Each run reports `Total time (mm:ss)` and `GPT tokens (in/out/total)`; both are written to
  `handwriting_result.json` (`elapsed`, `usage`, `cache_hits`). `--no-cache` bypasses,
  `--refresh-cache` recomputes.

### Golden dataset & regression checks

`golden/` holds the approved result for each sample. To make the samples reproduce their
approved golden **exactly and instantly**:

```bash
python3 seed_golden_cache.py     # seed the result-cache from golden/ (run once / after edits)
python3 compare_golden.py         # regression check: current output vs golden
python3 compare_golden.py --run   # re-run the 3 samples, then compare
```

After seeding, running the pipeline on any sample serves the golden in ~0.5 s. The
comparator fails only on **structural** regressions (missing/empty regions, count change)
or a **stable** field (dates, numbers, IDs, emails) changing; volatile cursive/`[?]` fields
are reported but never failed, since the model rewords them run-to-run.

`.cache/` is git-ignored. For portable reproducibility, commit `golden/` and run
`seed_golden_cache.py` on the new machine to rebuild the cache from golden.

## Notes & limitations

- **Handwriting is inherently ambiguous.** GPT may read an unclear date/name differently
  across runs — this is why results are cached (see above): once approved into `golden/`
  and seeded, a document reproduces the same reading every time. Treat *new* handwriting
  fields as *"needs review"*, especially where Content Understanding returns `null`.
- **Italic/cursive printed Hebrew is *printed*, not handwriting** — it belongs to the
  Content Understanding pipeline. The handwriting detector can occasionally over-detect
  italic script as handwriting; use `--italic` only as an opt-in when you want GPT to
  re-read italic (Content Understanding reads italic Hebrew less accurately).
- **Sensitive data:** the sample documents and generated results can contain personal
  information (names, ID numbers). `out/`, `output_handwriting/`, and `results/` are
  git-ignored; review the `sample/` files before pushing this repo to any remote.
- Content Understanding requires model deployments (`gpt-4.1` / `gpt-4.1-mini` /
  `text-embedding-3-large`) configured as defaults on the resource.

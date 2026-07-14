# Experiment Log

This is the durable record of OCR experiments in this repository. Update it for every meaningful
pipeline, prompt, model, preprocessing, or adjudication change, including failed experiments.
The goal is to remember not only what was tried, but whether it improved accuracy, cost, speed,
or reliability.

## Status Legend

- **ADOPTED**: enabled in the normal pipeline after measured improvement.
- **AVAILABLE**: retained as an opt-in mode or benchmark, but not the default.
- **REGRESSED**: measured worse than its comparison baseline.
- **INCONCLUSIVE**: insufficient evidence or mixed results; do not treat as an improvement.
- **REJECTED**: enough evidence exists not to use this approach.
- **INFRASTRUCTURE**: enables evaluation or operation but is not itself an accuracy result.
- **SUPERSEDED**: previously used, then replaced by a better or safer approach.

## Evaluation Rules

1. Golden/reference text is evaluation-only and must never be loaded during inference.
2. Record the exact source document/crop, model deployment, reasoning level, prompts, and image
   transformations.
3. Compare strict exact text, mean similarity, non-text classification, calls, tokens, and elapsed
   time when available. Wall-clock time alone is noisy.
4. Model-reported confidence is not calibrated confidence. Prefer independent agreement and
   deterministic checks.
5. Preserve failed and negative experiments. A result without a saved artifact is labeled
   **unverified**.
6. `analysis_results/`, `agent_benchmarks/`, and customer samples may contain PII and are ignored
   by Git. Paths in this log refer to local artifacts and may not exist on another clone.

## Current Decision Summary

- Normal customer-facing model: GPT-5.6 Sol low reasoning.
- Normal OCR shape: four core views in one multi-image call, followed by visual adjudication and
  deterministic guards.
- No persistent OCR, Content Understanding, or document-result cache.
- Golden data is never runtime input.
- Self-consistency voting remains opt-in.
- Word/glyph isolation is not adopted: it reduces cross-word semantic completion but also loses
  substantial Hebrew cursive accuracy.
- Candidate-blind disagreement handling remains the strongest next direction.

# Timeline

## 2026-07-08

### EXP-20260708-01: Initial two-engine extraction

- **Change**: Combined Azure AI Content Understanding for printed text/layout with GPT-5.4 vision
  for Hebrew handwriting.
- **Evidence**: commit `08562ef`.
- **Outcome**: Established the first working end-to-end extraction pipeline.
- **Decision**: **ADOPTED** architecture. This was a functional baseline, not a quality benchmark.

## 2026-07-09

### EXP-20260709-01: Exclude italic printed Hebrew from handwriting detection

- **Change**: Separated italic/cursive printed text from genuine handwriting.
- **Evidence**: commit `3c78a6b`.
- **Outcome**: Reduced printed-text false positives.
- **Decision**: **ADOPTED**.

### EXP-20260709-02: Golden regression baseline and overlapping CU execution

- **Change**: Added `golden/` reference results and `compare_golden.py`; started Content
  Understanding concurrently with handwriting work.
- **Evidence**: commit `5092fb1`.
- **Outcome**: Added repeatable structural/stable-field regression checks; CU overlap reduced
  avoidable wall-clock time without changing OCR evidence.
- **Decision**: **ADOPTED / INFRASTRUCTURE**.

### EXP-20260709-03: Fewer OCR views and confident-field early exit

- **Change**: Reduced per-image OCR versions and skipped visual reconciliation for sufficiently
  simple/agreed fields; added token and elapsed-time accounting.
- **Evidence**: commit `2769b86`.
- **Outcome**: Fewer calls with the same reasoning level. Exact historical A/B metrics were not
  preserved in a committed summary.
- **Decision**: **SUPERSEDED** by the later four-core-view multi-image design; token accounting
  remains **ADOPTED**.

### EXP-20260709-04: Stable versus volatile golden fields

- **Change**: Comparator distinguishes stable structured values from volatile cursive text.
- **Evidence**: commit `cae2daa`.
- **Outcome**: Prevented nondeterministic free text from failing structural regression checks while
  still reporting differences.
- **Decision**: **ADOPTED / INFRASTRUCTURE**.

### EXP-20260709-05: Persistent response/document caching

- **Change**: Added call cache, whole-document cache, and golden cache seeding.
- **Evidence**: commit `2a50313`, tag `working-consistent-cache`.
- **Outcome**: Made repeated inputs fast and byte-stable, but measured replay rather than fresh
  inference and risked coupling runtime behavior to golden results.
- **Decision**: **REJECTED / REMOVED**. Current policy requires fresh inference and no golden replay.

### EXP-20260709-06: Self-consistency voting

- **Change**: Multiple seeded OCR draws with deterministic aggregation. Structured fields use
  format-aware majority; free text uses a medoid/per-token vote.
- **Evidence**: commits `92f5402` and `2f6bacf`; `exp_determinism.py`.
- **Outcome**: Improved stability for dates, phones, IDs, and other structured values. Free-text
  voting could not recover omitted strokes or resolve severe visual ambiguity.
- **Decision**: **AVAILABLE**, opt-in with `VOTE=1`; not the default quality authority.

### EXP-20260709-07: Azure Document Intelligence `prebuilt-read`

- **Change**: Benchmarked deterministic OCR against GPT voting on the same handwriting crops.
- **Evidence**: commit `f8bcb49`; `azure_ocr_benchmark.py`.
- **Outcome**: 100% deterministic but inaccurate on Hebrew handwriting; reported similarity was
  roughly 0.1-0.4 and printed text was often confused with handwriting.
- **Decision**: **REJECTED** for Hebrew handwriting. Keep deterministic OCR for printed text only.

### EXP-20260709-08: Disable response cache by default

- **Change**: Returned benchmarking and normal runs to genuine computation.
- **Evidence**: commit `9794d6b`.
- **Outcome**: Restored valid fresh-run measurements.
- **Decision**: **SUPERSEDED** later by complete cache removal; fresh-only policy remains **ADOPTED**.

### EXP-20260709-09: Shorter per-call timeout

- **Change**: Reduced default read timeout to 90 seconds to limit tail latency.
- **Evidence**: commit `1d78804`.
- **Outcome**: Better bounded latency for GPT-5.4, but later Sol glyph experiments showed that
  complex outputs can exceed both 90 and 180 seconds.
- **Decision**: **ADOPTED** default with profile-specific overrides where measured.

### EXP-20260709-10: Smart image/PDF and multi-page input

- **Change**: Automatic image versus PDF handling, all-page defaults, page selection, and per-page
  output folders.
- **Evidence**: commit `a68ea8f`.
- **Outcome**: Expanded usable input surface without changing handwriting inference.
- **Decision**: **ADOPTED / INFRASTRUCTURE**.

## 2026-07-12

### EXP-20260712-01: InkSight review UI and fresh-only runtime

- **Change**: Added Flask review UI, upload/run history, live progress, diagnostics, page selection,
  and result inspection. Removed persistent OCR/document/golden replay behavior.
- **Evidence**: commit `32c38fd`; `app.py`, `ui/`, and `test_app.py`.
- **Outcome**: Human-inspectable evidence trail and genuine fresh analysis on each upload.
- **Decision**: **ADOPTED / INFRASTRUCTURE**.

### EXP-20260712-02: Seven-view quality checkpoint

- **Input**: 21-region human-verified `חיוב חשבון1` sample, GPT-5.4 medium.
- **Comparison**:

  | Pipeline | Exact | Similarity | Signature | Time | Calls | Tokens |
  |---|---:|---:|---:|---:|---:|---:|
  | Original seven-view baseline | 5/20 | 0.683 | 0/1 | 26:20 | 167 | 475,850 |
  | Quality-fixed seven-view | 5/20 | 0.744 | 1/1 | 18:53 | 168 | 496,472 |

- **Artifacts**: `analysis_results/חיוב חשבון1.pdf__2026-07-12_16-36-04/`,
  `analysis_results/חיוב חשבון1_quality_7view_v2/`.
- **Outcome**: Better similarity and signature handling, but similar exact accuracy and high cost.
- **Decision**: **SUPERSEDED** by core-view designs.

### EXP-20260712-03: Four independent core views

- **Result**: 7/20 exact, 0.746 similarity, 1/1 signature, 12:17, 105 calls, 341,970 tokens.
- **Artifact**: `analysis_results/חיוב חשבון1_core4_independent/`.
- **Outcome**: Best exact count among this A/B set, with materially lower cost than seven views.
- **Decision**: **AVAILABLE** with `OCR_MULTI_IMAGE=0`; quality/cost fallback.

### EXP-20260712-04: Four-view multi-image OCR

- **Result**: 6/20 exact, 0.767 similarity, 1/1 signature, 09:10, 44 calls, 177,818 tokens.
- **Artifact**: `analysis_results/חיוב חשבון1_core4_multi_resilient/`.
- **Outcome**: Versus original baseline, about 65% faster, 74% fewer calls, 63% fewer tokens, and
  higher exact/similarity scores. One fewer exact field than independent core views.
- **Decision**: **ADOPTED** default because it had the best measured overall quality/cost balance.

### EXP-20260712-05: Highlighted contextual rereads

- **Comparison**:

  | Experiment | Exact | Similarity | Signature | Time | Calls | Tokens |
  |---|---:|---:|---:|---:|---:|---:|
  | Fresh baseline | 6/20 | 0.749 | 1/1 | 16:16 | 45 | 188,224 |
  | Highlighted context v1 | 7/20 | 0.699 | 1/1 | 13:12 | 55 | 211,493 |
  | Conservative context v2 | 8/20 | 0.798 | 1/1 | 21:47 | 61 | 255,899 |
  | Spatial numeric cells | 7/20 | 0.814 | 1/1 | 12:15 | 61 | 252,607 |
  | Nearby CU reading order | 7/20 | 0.750 | 1/1 | 13:31 | 59 | 253,120 |

- **Artifacts**: `analysis_results/חיוב חשבון1_contextual_step1/`,
  `analysis_results/חיוב חשבון1_contextual_step1_v2/`,
  `analysis_results/חיוב חשבון1_spatial_numeric_step2/`, and
  `analysis_results/חיוב חשבון1_nearby_cu_step3/`.
- **Outcome**: Conservative v2 improved exact and similarity together. Spatial cells improved mean
  similarity but lost one exact field. Nearby CU context regressed because Markdown reading order
  is not a reliable coordinate map.
- **Decision**: Conservative highlighted reread **ADOPTED**; spatial cells **AVAILABLE** opt-in;
  nearby CU context **REGRESSED** and remains disabled.

### EXP-20260712-06: Reasoning-level speed/quality

- **Finding**: Medium reasoning preserved difficult cursive quality; low reasoning was faster but
  produced empty/garbled hard fields. More low-reasoning draws did not fully recover quality.
- **Decision**: Medium remained GPT-5.4 quality default. Later Sol-low evaluation changed the
  customer-facing model choice, not this GPT-5.4 finding.

## 2026-07-13

### EXP-20260713-01: GPT-5.6 Sol reasoning profiles

- **Artifacts**: `analysis_results/חיוב חשבון1.pdf__2026-07-13_07-10-13/`,
  `...07-16-03/`, `...17-01-49/`, `...17-08-12/`, plus `חיוב חשבון2` runs.
- **Outcome**: Sol-low gave a strong speed/token tradeoff but remained less reliable on ambiguous
  handwriting. Sol-medium was much slower on some runs. Exact cross-model metrics varied by
  pipeline revision, so individual run folders must not be compared without matching code/config.
- **Decision**: Sol-low eventually **ADOPTED** as customer UI default after repeat/consensus tests.

### EXP-20260713-02: GPT-5.6 Luna

- **Reported result**: 10 regions, 82 calls, 222,214 tokens, 04:19, macro similarity 0.373 on the
  human-golden document; over-segmented and misread key values.
- **Evidence quality**: Recorded in project notes; no single formal comparison summary is tracked.
- **Decision**: **REGRESSED**, experimental only.

### EXP-20260713-03: GitHub Copilot Claude Opus profiles

- **Artifacts**: `analysis_results/חיוב חשבון1.pdf__2026-07-13_10-17-26/` and
  `...10-26-06/`.
- **Outcome**: Faster in measured runs but did not outperform GPT-5.4 on the human-golden sample.
- **Decision**: **AVAILABLE** for controlled experiments; hidden from normal customer picker.

### EXP-20260713-04: GitHub Copilot Gemini 3.5 Flash

- **Reported result**: 11/20 exact, 81.81% similarity, 1/1 classification, 65 calls, 09:08 on
  `חיוב חשבון1`.
- **Artifact**: `analysis_results/חיוב חשבון1.pdf__2026-07-13_10-52-15/`.
- **Evidence quality**: Run artifacts exist; the reported aggregate score is not stored in a
  dedicated committed comparison file.
- **Outcome**: Beat the contemporaneous GPT-5.4 and Opus runs on this one document, but was not
  validated broadly enough for customer default.
- **Decision**: **AVAILABLE / INCONCLUSIVE** across documents.

### EXP-20260713-05: Contextual acceptance threshold

- **Change**: Configurable `CONTEXTUAL_RECHECK_MIN_CONFIDENCE`, default 0.60.
- **Reported replay**: Corrected region 8 `גבעתיים` to `בארותיים`; 8/20 to 9/20 exact and
  similarity 0.736 to 0.752 without a new model call.
- **Decision**: **ADOPTED**, but later region-02 analysis showed that a useful 0.58 reread can still
  be rejected. Threshold-only decisions remain brittle.

### EXP-20260713-06: Mistral Document AI

- **Outcome**: Failed Hebrew handwriting: omitted handwriting, copied printed text, and produced
  unrelated script on hard regions.
- **Evidence quality**: Project notes and probe outputs; no formal benchmark summary committed.
- **Decision**: **REJECTED** for runtime/UI.

### EXP-20260713-07: Repeated structured-field guard

- **Change**: Conservatively align equal-length identifier-like values differing by one digit only
  when confidence is at least 0.90 with a 0.10 margin; exclude dates and free text.
- **Reported replay**: Corrected `6252731` to `5252 731`, improving one strict field with no model
  calls.
- **Decision**: **ADOPTED** deterministic guard.

### EXP-20260713-08: Sealed hybrid and crop-agent benchmarks

- **Artifacts**: `agent_benchmarks/חיוב חשבון1__sealed-hybrid__*`,
  `agent_benchmarks/חיוב חשבון1.pdf__*__crop-agent-xhigh__*`,
  `sealed_hybrid_benchmark.py`, and `crop_agent_benchmark.py`.
- **Outcome**: Established leakage guards and immutable inference/evaluation separation. Some run
  folders lack a complete formal score summary.
- **Decision**: **INFRASTRUCTURE / INCONCLUSIVE** as a winning OCR configuration.

### EXP-20260713-09: Three-run Sol-low spatial consensus

- **Result**: Independent runs scored 11/20, 8/20, and 9/20 exact. Deterministic consensus scored
  10/20 exact, 84.7% similarity, and 100% non-text classification.
- **Artifact**: `agent_benchmarks/חיוב חשבון1__sol-consensus/`; `sol_consensus.py`.
- **Outcome**: Did not beat the best single run on exact count, but improved repeatability and
  overall similarity without a reduction-stage model call.
- **Decision**: **AVAILABLE** benchmark; evidence supporting Sol-low as normal UI default.

## 2026-07-14

### EXP-20260714-01: Direct blind Sol-high crop OCR

- **Input**: Original region crops only; no golden/reference/context metadata.
- **Artifact**: `analysis_results/חיוב חשבון1__direct-sol-high-image-only__2026-07-14/`.
- **Outcome**: Region 02 still hallucinated `סניף פתח תקווה`, proving the phrase was not leaked by
  the app's printed context or golden data.
- **Decision**: **REGRESSED** as a grounded OCR strategy; important diagnostic result.

### EXP-20260714-02: Stroke-grounded and two-prompt Sol-high tests

- **Artifacts**: `analysis_results/region_02__sol-high-stroke-grounded-v2__2026-07-14/` and
  `analysis_results/region_02__sol-high-two-prompt-consistency__2026-07-14/`.
- **Outcome**: Stronger anti-invention prompts did not reliably prevent familiar-entity completion.
- **Decision**: **INCONCLUSIVE / not adopted**.

### EXP-20260714-03: Azure Sol versus Copilot vision request parity

- **Artifact**: `analysis_results/region_02__azure-sol-vs-copilot-gpt55__2026-07-14/`.
- **Controls**: Same prompt and image hashes; neither caller supplied an output limit.
- **Caveat**: Different models were available, so this is transport/backend comparison, not a
  controlled model-quality comparison.
- **Decision**: **INCONCLUSIVE** for OCR accuracy.

### EXP-20260714-04: App run on uploaded region 02

- **Artifact**: `analysis_results/01_original_crop.png__2026-07-14_10-35-43/`.
- **Trace**:
  1. Initial core4 candidate hallucinated `סניף פתח תקווה`.
  2. Visual reconciliation received that candidate and repeated it.
  3. Candidate-blind contextual reread returned `סניף 4 פארק הירקון` at confidence 0.58.
  4. The 0.60 gate retained the bad first result and only marked disagreement for review.
- **Outcome**: Identified candidate anchoring plus brittle confidence-gate behavior.
- **Decision**: **REGRESSED** decision path; motivates candidate-blind disagreement handling.

### EXP-20260714-05: GPT-5.4 line isolation

- **Method**: Three line crops, two candidate-blind reads per line, character alignment in Python.
- **Result**: 6 calls, 30,588 tokens. Removed `פתח תקווה` but produced heavy disagreement and mostly
  `[?]` consensus.
- **Artifact**: `analysis_results/region_02__gpt54-segmented-lines__2026-07-14/`.
- **Decision**: **REJECTED** as default; honest but insufficiently accurate.

### EXP-20260714-06: GPT-5.4 isolated visual groups

- **Method**: Three stable groups from line 2, two blind reads each.
- **Result**: 6 calls, 20,325 tokens. Reads included `בארץ`/`הארץ` and `הרצל`/`הררא`; did not
  recover `פארק הירקון`.
- **Artifact**: `analysis_results/region_02__gpt54-isolated-words__2026-07-14/`.
- **Decision**: **REJECTED** as default.

### EXP-20260714-07: Sol-low truly isolated visual groups

- **Method**: Six sequential calls; each request contained one visual group and no neighboring
  word, previous answer, or summary call.
- **Result**:

  | Group | Read A | Read B | Python consensus |
  |---|---|---|---|
  | 1 | `סניף` | `ס[?][?][?]` | `ס[?]` |
  | 2 | `באר שבע` | `באר[?].` | `באר[?]` |
  | 3 | `הרר[?][?]ל` | malformed JSON with raw uncertain glyphs | `[?]` |

- **Cost**: 6 calls, 11,425 tokens.
- **Artifact**: `analysis_results/region_02__sol-low-truly-isolated-words__2026-07-14/`.
- **Outcome**: Cross-word context was removed, yet Sol still mapped ambiguous local strokes to
  familiar Hebrew words. Isolation does not turn a multimodal LLM into a glyph classifier.
- **Decision**: **REJECTED** as default.

### EXP-20260714-08: Forced four-unit gap split

- **Hypothesis**: Split line 2 into `סניף | 4 | פארק | הירקון` before OCR.
- **Preview result**: Join width 17 produced four boxes, but split the left word `הירקון` into two
  pieces while the right group still combined `סניף` with a large adjacent mark.
- **Artifact**: `analysis_results/region_02__sol-four-unit-multiview-preview__2026-07-14/`.
- **Decision**: **REJECTED before model calls**. Geometry was visibly invalid, so no tokens were
  spent. This is why preview gates are mandatory.

### EXP-20260714-09: Three-group faithful multi-view Sol-low

- **Hypothesis**: Preserve the three stable isolated groups, but provide original-with-form-line,
  blue-only, and enlarged/sharpened views of the same group in each request.
- **Controls**: No neighboring handwriting; two blind prompts; no previous candidate; sequential
  Sol-low calls; 2,500 output-token cap.
- **Artifacts**: Preview at
  `analysis_results/region_02__sol-three-group-multiview-preview__2026-07-14/`; result target at
  `analysis_results/region_02__sol-low-three-group-multiview__2026-07-14/`.
- **Operational result**: The first sequential run with a 2,500-token cap timed out on the Sol
  endpoint after 240 seconds before producing a complete result.
- **Bounded probe**: Middle group only, prompt A, 1,800-token cap, completed as `בארץ`; 1 call,
  1,410 total tokens. Artifact:
  `analysis_results/region_02__sol-three-group-multiview-preview__2026-07-14/sol_low_word_02_probe.json`.
- **Paired result**:

  | Stable visual group | Read A | Read B | Python consensus |
  |---|---|---|---|
  | 1 | `סניף` | `סניף` | `סניף` |
  | 2 | `פארק` | `פארק` | `פארק` |
  | 3 | `היררל[?][?][?]` | `הי [?].[?][?][?]ל` | `הי[?][?][?][?][?]` |

- **Cost**: 6 sequential calls, 4,602 input tokens, 6,819 output tokens, 11,421 total.
- **Artifact**:
  `analysis_results/region_02__sol-low-three-group-multiview-bounded__2026-07-14/result.json`.
- **Comparison**: The same three groups with one blue-only view produced
  `ס[?] באר[?] [?]`. Multi-view evidence recovered `סניף` and `פארק` exactly and the `הי`
  prefix of `הירקון`. It did not recover the visible digit `4` or the remainder of `הירקון`.
- **Interpretation**: Multiple faithful views materially improved this crop and prevented the
  `באר שבע`/`בארץ` substitution in the paired run. The isolated one-prompt probe still returned
  `בארץ`, so two independent prompts remain necessary.
- **Decision**: **AVAILABLE / PROMISING**, not adopted from a single crop. Next validation must run
  this configuration across multiple human-golden regions and compare exact accuracy, not only
  region 02.

### EXP-20260714-10: Opt-in InkSight strategy integration

- **Change**: Added an **OCR strategy** selector to InkSight. The experimental option supports
  GPT-5.4 medium and GPT-5.6 Sol low, and persists in `run.json`, live snapshots, and saved-run
  labels. Other model profiles are rejected by the API for this strategy.
- **Runtime behavior**: Valid colored lined regions use adaptive isolation. Rows with 2-4 stable
  groups use per-group multi-view reads; over-segmented rows stay intact as one isolated line.
  Every unit gets original-with-form-line, blue-only, and enlarged/sharpened views, followed by two
  candidate-blind calls and Python alignment. At most two draws run concurrently. Invalid or failed
  regions fall back to standard OCR. Stable grouped lines are scheduled first. GPT-5.4 uses a
  larger output budget than Sol to avoid empty reasoning-only responses and duplicate failover.
- **Decision behavior**: Experimental consensus bypasses candidate-fed visual reconciliation and
  is marked `isolated_multiview_consensus`; disagreements remain `[?]` and `needs_review`.
- **Status**: **AVAILABLE**, opt-in only. Standard core4 remains the default.
- **Validation**: Offline pipeline integration writes normal `handwriting_result.json`, region
  diagnostics, merged overlay artifacts, and never invokes visual reconciliation for a successful
  experimental plan.

### EXP-20260714-11: GPT-5.4 isolation run exposed unsealed merge

- **Artifact**: `analysis_results/01_original_crop.png__2026-07-14_16-04-56/`.
- **Model/config**: GPT-5.4 medium, 6,000 output-token cap, two concurrent draws, 120-second
  per-endpoint timeout.
- **Raw isolation result**: 9/10 draws completed and one whole-line read failed. The two blind
  candidates were highly uncertain, and Python agreement produced
  `[?]\nס[?]ל כא[?] ה[?][?]ל\nהרב[?] [?][?][?][?]`.
- **Cost**: 10:31 elapsed, 10,012 input tokens, 31,633 output tokens, 41,645 total usage tokens,
  and 12 recorded GPT usage events including detection and merge.
- **Unexpected merged result**: `merged.md` contained fluent text including `בנק הפועלים` and
  `סניף פארק הירקון`, despite neither isolated candidate supporting that reconstruction.
- **Root cause**: This was **not golden-data leakage**. `handwriting_pipeline.py` does not read the
  golden dataset. The app's golden endpoint reads golden files only for post-run comparison.
  However, `build_merged_markdown` made an additional unsealed vision call with the original page,
  Content Understanding output, and isolated candidates. CU already contained close but garbled
  forms such as `בנק הבו180ם`, `סני4 פארק ה׳רהון`, and `הרצא אמר`; the merge model reread the page
  and semantically repaired them.
- **Interpretation**: The fluent `merged.md` is a contextual full-page reconstruction, not evidence
  that isolated OCR succeeded. It must not be scored as the isolation result.
- **Historical controls**:
  - In `analysis_results/01_original_crop.png__2026-07-14_10-35-43/`, standard Sol OCR supplied the
    confident but wrong candidate `סניף פתח תקווה`; the semantic merge preserved that wrong anchor.
  - In `analysis_results/01_original_crop.png__2026-07-14_10-46-15/`, all standard Sol handwriting
    calls failed and supplied an empty candidate; the same full-page merge independently recovered
    `סניף פארק הירקון` from CU plus the image, with only the final word still wrong.
  - The isolated run used GPT-5.4 rather than Sol, so model choice is an additional confound.
- **Causal conclusion**: Isolation may have helped only negatively by replacing a confident wrong
  anchor with uncertainty. It supplied no demonstrated positive evidence for the recovered words,
  and full-page reconstruction had already succeeded without isolation when no bad candidate was
  present.
- **Decision**: **REGRESSED for experimental evaluation**. Preserve `handwriting_result.json` as
  the authoritative isolated result. A future change should disable semantic full-page merge for
  isolated mode or label it as a separate, non-isolated reconstruction.

### EXP-20260714-12: GPT-5.4 standard fast-profile full-document run

- **Input**: The 21-region human-golden `חיוב חשבון1.pdf` sample.
- **Artifact**: `analysis_results/חיוב חשבון1.pdf__2026-07-14_17-36-52/`.
- **Model/config**: GPT-5.4 standard core4 strategy, low OCR and detection reasoning, 4,000 OCR
  output-token cap, 2,500 detection cap, one endpoint per attempt, 60-second read timeout, and no
  final original-image rescue.
- **Raw OCR result**: 21/21 draws completed with no failed regions. Post-run comparison of
  `handwriting_result.json` reported 40.0% exact text, 75.3% mean similarity, and 100.0% non-text
  classification. The exact score is 8/20 text regions; the signature is scored separately.
- **Cost and time**: 04:29 pipeline time, 63 GPT calls, 146,150 input tokens, 37,501 output tokens,
  and 183,651 total tokens.
- **Telemetry caveat**: The live snapshot near the end of OCR showed only 22 calls and 51,203
  tokens, while final artifact parsing reported 63 calls and 183,651 tokens. Live usage is
  therefore incomplete during concurrent work and must not be treated as final accounting.
- **Merge caveat**: The job remained visibly in `merging` after region OCR completed, but the
  current artifacts do not retain stage timestamps, so an exact semantic-merge duration cannot be
  recovered. `merged.md` remains contextual reconstruction and was not used for the accuracy
  figures above.
- **Comparison caveat**: This run is faster than historical GPT-5.4 core4 runs, but pipeline and
  prompt revisions differ, so it is not a controlled A/B against those artifacts.
- **Decision**: **AVAILABLE / PROMISING for latency**. Preserve as the GPT-5.4 standard fast
  profile while gathering matched repeated runs before making a quality claim.

### EXP-20260714-13: GPT-5.4 fast profile on second document

- **Input**: The 15-region human-golden `חיוב חשבון2.pdf` sample.
- **Artifact**: `analysis_results/חיוב חשבון2.pdf__2026-07-14_17-44-15/`.
- **Model/config**: Same GPT-5.4 standard fast profile as EXP-20260714-12.
- **Raw OCR result**: Detection found 15 regions. All initial draws completed, one empty region was
  retried, and no draw failed. Post-run comparison of `handwriting_result.json` reported 14.3%
  exact text, 59.8% mean similarity, and 0.0% non-text classification. The exact score is 2/14
  text regions; the signature region was returned as uncertain handwritten text rather than
  `signature_non_text`.
- **Cost and time**: 05:33 pipeline time, 48 GPT calls, 125,062 input tokens, 30,522 output tokens,
  and 155,584 total tokens.
- **Observed failure modes**: Digit spacing and substitution errors affected account/reference
  numbers; difficult names, addresses, and margin notes were incomplete or incorrect; one colored
  handwriting region was empty; and the signature classification failed.
- **Decision**: **REGRESSED for accuracy on this document**. The fast profile remains useful for
  latency experiments, but the two-document evidence does not support promoting it as a quality
  default. Continue to score raw OCR rather than semantic merged output.

### EXP-20260714-14: Optional detection repair caused two document failures

- **Input**: `חיוב חשבון2.pdf`.
- **Failed attempts**:
  - GPT-5.4 isolated multi-view:
    `analysis_results/חיוב חשבון2.pdf__2026-07-14_17-53-54/`.
    The geometry-repair response was empty.
  - GPT-5.6 Sol-low standard:
    `analysis_results/חיוב חשבון2.pdf__2026-07-14_18-16-20/`.
    The geometry-repair response exceeded the 90-second read timeout.
- **Root cause**: Both primary detection calls returned parseable regions with a geometry issue.
  The optional geometry-repair call then raised `RuntimeError`, which escaped
  `detect_handwriting_regions` and failed the whole page. These were repair-stage failures, not
  evidence that the models could not detect handwriting on this document.
- **Change**: Geometry repair and the optional lower-page coverage pass now catch model-runtime
  failures, emit a visible diagnostic, and retain the regions already detected. Primary detection
  remains required; only supplemental calls are best-effort.
- **Verification**: A detection-only GPT-5.4 fast-profile run on the exact rendered page completed
  with 15 regions and improved geometry penalty from 10.6 to 0.0. The full suite passes 131 tests,
  including explicit empty-output and timeout regression cases. A subsequent full InkSight
  GPT-5.4 isolated run, `חיוב חשבון2.pdf__2026-07-14_18-28-43`, also found 15 regions, improved
  geometry penalty from 11.3 to 0.0, and entered handwriting OCR instead of failing detection.
- **Accounting caveat**: Failed/empty responses are not represented in successful usage events, so
  the failed GPT-5.4 job's displayed zero calls and tokens was not its true request cost.
- **Decision**: **ADOPTED / INFRASTRUCTURE**. The two runs remain inconclusive for isolated OCR
  accuracy because they never reached OCR, but the shared detection failure is fixed.

### EXP-20260714-15: GPT-5.4 isolated multi-view on second document

- **Input**: The 15-region human-golden `חיוב חשבון2.pdf` sample.
- **Artifact**: `analysis_results/חיוב חשבון2.pdf__2026-07-14_18-28-43/`.
- **Model/config**: GPT-5.4 isolated multi-view. Only regions 4 and 9 produced valid isolated
  plans; the other 13 regions fell back to standard OCR because segmentation requirements were
  not met. The run planned 27 draws, made three empty-region retries, and recorded two timed-out
  standard core4 attempts.
- **Raw OCR result**: Post-run comparison of `handwriting_result.json` reported 14.3% exact text,
  61.2% mean similarity, and 0.0% non-text classification. The exact score is 2/14 text regions;
  the signature was returned as uncertain handwritten text.
- **Cost and time**: 27:01 pipeline time, 57 GPT calls, 117,085 input tokens, 145,041 output tokens,
  and 262,126 total tokens.
- **Same-document standard comparison**: EXP-20260714-13 scored the same 2/14 exact and 0.0%
  classification with 59.8% similarity, 05:33, 48 calls, and 155,584 tokens. Isolation improved
  similarity by only 1.4 percentage points while taking 4.9 times longer and using 68% more tokens.
- **Decision**: **REGRESSED**. This document provides no evidence that isolated multi-view improves
  exact accuracy or classification, and its small similarity increase does not justify the severe
  latency and token cost. Keep it experimental and opt-in only.

### EXP-20260714-16: GPT-5.4 standard fast-profile repeat on second document

- **Input**: The same 15-region human-golden `חיוב חשבון2.pdf` sample.
- **Artifact**: `analysis_results/חיוב חשבון2.pdf__2026-07-14_19-07-26/`.
- **Model/config**: Same GPT-5.4 standard fast profile as EXP-20260714-13.
- **Raw OCR result**: 21.4% exact text, 52.7% mean similarity, and 0.0% non-text classification.
  The exact score is 3/14 text regions; the signature was again classified as handwritten text.
- **Cost and time**: 06:28 pipeline time, 49 GPT calls, 127,748 input tokens, 29,773 output tokens,
  and 157,521 total tokens. All 15 initial draws completed, one empty region was retried, and no
  draw ultimately failed.
- **Repeatability**: The earlier same-profile run scored 2/14 exact and 59.8% similarity. The repeat
  gained one exact field but lost 7.1 similarity points, confirming substantial output variance on
  this document despite similar cost and runtime.
- **Decision**: **REGRESSED / INCONCLUSIVE for quality**. The fast profile remains operationally
  useful, but neither repeat supports it as a quality default and the two runs are not stable enough
  to infer small accuracy improvements.

# Experiment Template

Copy this section for every new experiment.

```markdown
### EXP-YYYYMMDD-NN: Short name

- **Status**: RUNNING | ADOPTED | AVAILABLE | REGRESSED | INCONCLUSIVE | REJECTED
- **Question/Hypothesis**:
- **Input and reference**:
- **Inference isolation**: What context, candidates, metadata, and golden data were unavailable?
- **Model/config**: Deployment, reasoning, seeds, timeout, output cap, concurrency.
- **Image evidence**: Original/enhanced crops and segmentation settings.
- **Baseline**:
- **Result**: Exact, similarity, classification, calls, tokens, elapsed time.
- **Artifacts**:
- **Observed failure modes**:
- **Decision and next step**:
```

# Open Questions

1. Can candidate-blind visual adjudication use full faithful line context without seeing the first
   OCR candidate, then resolve disagreement deterministically or return `unresolved`?
2. Can CU polygon geometry provide safe nearby printed context, replacing unreliable Markdown
   reading-order windows?
3. Can independent-source agreement (CU, blind visual reread, alternate model) outperform
   self-reported confidence without introducing semantic leakage?
4. Should malformed model JSON be repaired deterministically before character alignment, while
   preserving the raw response?
5. How well do successful settings generalize beyond `חיוב חשבון1`? A setting is not a new default
   until it is measured across multiple document types.

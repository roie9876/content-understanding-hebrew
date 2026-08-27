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
- Deterministic postprocessing is measured with `quality_replay.py`; CU evidence remains separate
  from model-facing context.
- High reasoning is not a general fallback. It is restricted to explicit uncertainty-only Hebrew
  text after signature rejection.

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
- **2026-07-15 UI integration**: The existing account-specific Copilot SDK backend is exposed as
  `Claude Opus 4.8 · GitHub Copilot SDK · Core4`. The profile explicitly enables the standard
  four-view multi-image Core4 call, high reasoning, visual reconciliation, and the existing
  180-second timeout. The isolated-multiview strategy remains incompatible and selecting Claude
  switches the UI back to `standard`.
- **Fresh integrated validation**: `חיוב חשבון1.pdf__2026-07-15_18-00-19` completed through the
  real SDK bridge with 21/21 detected regions, 67 Copilot calls, and 03:18 elapsed. It scored
  7/20 strict exact text, approximately 0.75 mean region similarity, and 1/1 non-text
  classification. Exact fields included `מרטין פאלרמו`, `5298570`, `4571130`, `בארותיים`,
  `ביוב`, and `550`; important date/account errors remain. The configured account advertised
  exact model ID `claude-opus-4.8` with vision enabled and policy state `enabled`.
- **Decision**: **AVAILABLE** as an explicit high-cost experimental picker option; not the default.

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

### EXP-20260714-17: Black-ink ruled-form crop tightening

- **Status**: ADOPTED
- **Question/Hypothesis**: The oversized crops on `דוח_אלעד.pdf` persist because the black-ink
  refiner can only expand a model box. Long form rules plus unusually tall connected components
  should identify the actual handwritten row without another model call.
- **Input and reference**: Saved pages and detections from
  `analysis_results/דוח_אלעד.pdf__2026-07-14_19-31-00/` and the cancelled repeat
  `analysis_results/דוח_אלעד.pdf__2026-07-14_20-05-05/`.
- **Inference isolation**: Offline OpenCV replay only. No model, OCR candidates, merged output, or
  golden text was used to choose geometry.
- **Baseline**: Six overlapping crops were 365-497 pixels tall. Region 5 included two income rows,
  and region 1 contained mostly printed prose while clipping the intended handwriting at its lower
  edge. The existing `refine_box_to_ink` preserved the complete original box and added vertical
  padding, so it could not correct either failure.
- **Failed first fix**: Local ruled-row tightening worked offline when GPT returned one approximate
  box per handwriting row. On the next run GPT returned only four boxes, including two prose/table
  containers and omitting two handwritten rows. A local postprocessor cannot reject a false box
  with only one nearby rule or create rows that the model never returned.
- **Adopted change**: For monochrome pages with at least three strongly supported ruled rows, scan
  the full page for long horizontal boundaries and select intervals with at least six tall stroke
  components, sufficient ink area, and a dense horizontal handwriting core. This replaces the
  model geometry and uses generic row labels. The prior local tightener remains the fallback for
  pages without enough global evidence. Colored-ink and paragraph-block behavior is unchanged.
- **Exact repeat replay**: The failed repeat's four boxes included prose at `(871,1510,1735,332)`
  and a work-to-table container at `(801,2484,1845,418)`. Global replay returns six boxes:
  `(921,1958,1753,134)`, `(637,2195,2067,155)`, `(780,2504,869,158)`,
  `(1077,2997,1305,132)`, `(989,3181,1303,88)`, and `(944,3391,1419,94)`.
  No prose box remains, both omitted rows are restored, and sparse printed checkbox/table outliers
  are excluded from the horizontal bounds.
- **Verification**: Three synthetic regressions cover local snapping, distinct assignment for
  overlapping detections, and global recovery when model detection misses rows. A hard replay
  assertion on the exact failed page verifies six expected y-bands and tight work/table widths.
  The complete suite passes 134 tests; VS Code reports no errors in the edited files.
- **Decision**: **ADOPTED / GEOMETRY**. Use global ruled-row discovery when its conservative
  evidence gate succeeds; otherwise preserve the local fallback.

### EXP-20260714-18: HASH v1 advisory OCR sidecar

- **Status**: AVAILABLE / LIVE VALIDATED
- **Question/Hypothesis**: A separately trained CNN-BiLSTM-CTC Hebrew model may expose useful
  letter or spelling evidence when Core4 and GPT visual adjudication disagree, even though its
  standalone natural-handwriting accuracy is insufficient.
- **Source**: `roie9876/heb-handfree` commit `b63051c`, cloned to `~/heb-handfree`. The bundled
  HASH v1 checkpoint is served by FastAPI through `POST /v1/ocr`; `GET /health` reports readiness.
- **Measured upstream limitation**: The model card reports 45.95% CER and 100% WER on its confirmed
  natural ruled-form evaluation. Confidence is mean greedy CTC path probability and uncalibrated.
- **Integration**: Check sidecar health once per page, submit each original region crop while Core4
  runs, persist raw text/line/confidence/box evidence in `region_*/ivrit_ocr.json`, and include one
  `ivrit_hash_v1` candidate in final GPT visual adjudication. HASH is explicitly excluded from
  deterministic agreement, faithful-source guards, semantic fallback, and standalone acceptance.
  Absence, disablement, HTTP errors, and timeouts are nonfatal.
- **UI**: Added an **Ivrit OCR** diagnostics tab with per-region status, advisory text,
  uncalibrated confidence, and crop-relative line boxes. Completed evidence is also retained in
  `result.json`, `handwriting_result.json`, and `ocr_progress.json`.
- **Validation**: Mocked API contract, prompt-safety, and full one-region orchestration tests pass.
  The complete 136-test suite also exercises a real localhost connection-refused path and confirms
  the page continues normally. JavaScript syntax, rendered tab declaration, editor diagnostics,
  and `git diff --check` pass. Homebrew `uv 0.11.28` provisioned the upstream frozen Python 3.12
  environment. The FastAPI service loaded HASH v1 with Apple MPS, returned `{"status":"ok"}` from
  `/health`, and completed a real production-adapter request against a saved InkSight crop. That
  crop produced two low-quality lines at 0.8523 uncalibrated mean confidence, reinforcing that HASH
  must remain advisory despite high model confidence.
- **Decision**: Keep enabled as an optional localhost advisory source. Do not treat HASH confidence
  as acceptance confidence or promote its text without visible-stroke support from GPT.

### EXP-20260715-19: HASH contribution ablation and selective gate

- **Status**: ADOPTED / SELECTIVE ADVISORY ONLY
- **Question**: Does adding HASH v1 to GPT-5.4 Core4 improve final transcription, and can weak
  evidence be filtered without golden data at runtime?
- **Inputs**: Two complete 21-region runs of the human-golden `חיוב חשבון1.pdf`, plus paired visual
  reconciliations using immutable images and Core4 responses from
  `חיוב חשבון1.pdf__2026-07-14_21-39-20`. Golden text was loaded only after ablation inference.
- **Observed full-run scores**: Always-on HASH runs scored 5/20 exact at 0.759 similarity and 6/20
  exact at 0.739. The nearest same-profile no-HASH run scored 8/20 at 0.753. Run-level differences
  are not causal because Core4 and visual adjudication vary between calls.
- **HASH reliability**: HASH was byte-identical on 21/21 regions across both runs, but standalone
  accuracy was only 2/20 exact with 0.2645 mean similarity. Every text field had confidence above
  0.90; even the subset above 0.95 remained only 2/11 exact per run. HASH confidence is therefore
  repeatable but unusable as correctness evidence.
- **Ungated paired test**: Five representative regions x two repetitions produced 0/10 exact and
  0.6481 similarity without HASH versus 2/10 and 0.6679 with HASH. However, both exact gains were
  on `ביוב` while HASH itself incorrectly read `כסף`, showing prompt perturbation rather than valid
  OCR support. The three-line recipient field worsened in both HASH pairs. Numeric/date pairs were
  unchanged. Artifact: `analysis_results/ivrit_hash_ablation_sequential__2026-07-15/`.
- **Selective gate**: Admit HASH only for single-line nonnumeric/non-structured Hebrew, at least
  three letters per source, at least 60% length overlap, and at least 40% normalized Hebrew
  character agreement with Core4. This admitted only regions 8/9 in one run and 8/9/19 in the
  other, versus all 21 previously. It excludes HASH confidence from the GPT payload and omits all
  HASH-specific prompt text when the gate fails. A follow-up live run exposed a noisy detector
  label (`handwritten number`) on exact Hebrew `בארותיים`; generic number labels no longer veto
  nonnumeric readings, while exact Core4-HASH duplicates are omitted as redundant.
- **Gated paired test**: Three eligible regions x two repetitions produced 0/6 exact and 0.6056
  similarity without HASH versus 1/6 exact and 0.7056 with gated HASH. No pair regressed. HASH
  rescued exact `בארותיים` in one of two repetitions; the other eligible fields were unchanged.
  The benefit is real but stochastic, so HASH remains a hint rather than an authority. Artifact:
  `analysis_results/ivrit_hash_gated_ablation__2026-07-15/`.
- **Invalid attempt**: The first 20-call concurrent ablation is excluded because all calls failed
  at TCP connect during a transient Foundry endpoint outage. Sequential retry completed without
  failures after DNS/TLS and Entra checks passed.
- **Validation**: 139 Python tests, JavaScript syntax, editor diagnostics, and `git diff --check`
  pass. Live InkSight reports HTTP 200, HASH `/health` reports `ok`, and the adapter exposes the
  configured 0.40 gate threshold. The UI identifies both admitted and excluded HASH evidence.
  A subsequent live 21-region run admitted only two HASH readings and scored 7/20 exact, 0.7902
  similarity, and 1/1 classification. This exceeds both always-on HASH runs in similarity, but
  remains observational because Core4/visual outputs vary between runs. The live artifact also
  exposed a noisy `handwritten number` label on Hebrew `בארותיים`; the final gate ignores generic
  number labels when readings are nonnumeric and suppresses exact Core4-HASH duplicates.
- **Eight-document deterministic ablation**: Replaying the final selective HASH replacement over
  all 70 human text regions changed only three finals, all on `חיוב חשבון1`. Exact text improved
  from 29/70 without HASH to 31/70 with HASH, and strict normalized similarity improved from
  0.687 to 0.704. It recovered exact `מרטין פאלרמו` and `בורוכוב`; the third field improved from
  0.333 to 0.667 similarity without becoming exact. No exact field regressed. Only 3/77 emitted
  regions were eligible, confirming that HASH is a narrow spelling corrector rather than a general
  OCR authority.
- **Operational cost**: A warm local MPS benchmark processed all 21 `חיוב חשבון1` crops in 0.99s
  total (35ms median, 104ms p95, 165ms maximum). Calls overlap GPT work, so normal minutes-long
  page latency is effectively unchanged. The meaningful cost is maintaining the separate local
  Python 3.12/FastAPI/model service; a failed health check remains nonfatal.
- **Decision**: **ADOPTED** with the deterministic selective gate. Keep all raw HASH evidence in
  artifacts/UI, never expose its confidence to GPT, and never use HASH as a standalone fallback.
  Keep it default-on in this quality-first local environment, but optional and fail-open rather
  than a required deployment dependency.

### EXP-20260715-20: Cross-Mac HASH discrepancy and crop normalization

- **Status**: ADOPTED / TARGETED PREPROCESSING
- **Reported discrepancy**: The source Mac returned exact `מרטין פאלרמו` from a manually prepared
  tight sample. InkSight region 5 returned `משת פאלר מן\nן` from the same document content.
- **Device check**: The exact InkSight PNG (`704x170`, SHA-256
  `141119b67b556ec795a0908187ea130dea3c25551e2c758839701ab67bc1b44a`) produced identical text,
  boxes, and effectively identical confidence on CPU and Apple MPS. The checkpoint/device is not
  the cause. The source sample file is ignored/unavailable in the cloned repository, so its byte
  hash could not be compared; the screenshot visibly shows a tighter one-line crop.
- **Root cause**: InkSight's original crop includes a black rule and printed field label. HASH
  segmented one handwritten row into two vertically overlapping lines, including a duplicate final
  `ן`. Removing black print and retaining only tight blue pen pixels produced exact
  `מרטין פאלרמו` through the live MPS service.
- **Global preprocessing benchmark**: Across 20 text regions, original HASH scored 2/20 exact and
  0.2645 similarity; blue-only scored 2/20 and 0.2665; rule-clipped scored 1/17 and 0.2589. A global
  switch is rejected. Blue-only exactly fixed region 5 and improved regions 19/20, but degraded
  region 9 and the multiline region 2. Artifact:
  `analysis_results/hash_input_preprocessing__2026-07-15/`.
- **Adopted change**: Retry blue-only only when Core4 is one nonnumeric Hebrew line and original
  HASH has multiple lines with at least 50% vertical overlap. Select blue-only only if it becomes
  one line and agrees with original HASH by at least 55% characters and 80% length. This
  preprocessing consensus may admit spelling evidence even when Core4 is the outlier. Both variant
  summaries and `selected_variant` are persisted and displayed in the Ivrit OCR tab.
- **Verification**: The production helper plus live MPS service returns exact `מרטין פאלרמו` for
  the saved region 5 crop. Focused helper and full orchestration regressions verify two HASH calls,
  blue-only selection, persisted preprocessing consensus, and delivery to visual reconciliation.
- **Decision**: **ADOPTED** as a narrow duplicate-line recovery path. Keep original HASH input for
  all normal, multiline, structured, and non-overlapping cases.

### EXP-20260715-21: Contextual override versus cross-engine agreement

- **Status**: ADOPTED / SEMANTIC-ANCHORING GUARD
- **Failure**: On region 8, Core4 read `בארותיים`, visual reconciliation kept `בארותיים`, and HASH
  independently read `בארותיים`. The later highlighted contextual reread noticed the printed field
  label `עיר` (city), guessed `אורנים` at confidence 0.69, and replaced the agreed result because
  the generic contextual acceptance floor was 0.60. The app marked disagreement for review but
  still chose the context guess.
- **Root cause**: `accept_contextual_recheck` used information length, script, structured-token
  preservation, and self-reported confidence, but had no protection for exact independent source
  agreement. Printed semantic context could therefore anchor a plausible replacement.
- **Change**: If the pre-context visual result and raw HASH text agree exactly after normalization,
  and Core4 either agrees exactly or supplied the measured correlation that admitted HASH through
  the selective gate, a differing contextual reread cannot replace them. The contextual text
  remains an alternative, the region is marked `needs_review`, and `cross_engine_agreement`
  records exact/correlated sources and rejected context text. Normal contextual correction remains
  unchanged when HASH is absent, unrelated, or ineligible.
- **Exact replay**: Saved region 8 now selects `בארותיים`, retains `אורנים` and `אורנית[?]` as
  alternatives, and records `cross_engine_agreement_protected` instead of `contextual_recheck`.
- **Validation**: Exact-artifact replay and focused positive/control regressions pass. The complete
  suite passes 142 tests; JavaScript syntax, editor diagnostics, and `git diff --check` pass.
- **Decision**: **ADOPTED**. Nearby printed context remains useful, but one context reread cannot
  overrule exact agreement from visual/Core4 plus independent HASH.

### EXP-20260715-22: Region 4 blue-ink noise isolation

- **Status**: INCONCLUSIVE / SEGMENTATION CONFIRMED
- **Question**: Does removing printed Hebrew, horizontal form rules, and distant handwriting make
  HASH reliable on the noisy multiline region 4 from `חיוב חשבון2.pdf`?
- **Input**: Exact `1158x352` production crop from run
  `חיוב חשבון2.pdf__2026-07-15_10-07-50`. Original HASH produced three heavily overlapping lines:
  `קך רב ת`, `הת אדז במתג`, and `ל`.
- **Isolation**: Retained only blue-dominant pen pixels, removed connected components below eight
  pixels, tested color and black-on-white renderings, bridged small rule-crossing gaps with `3x3`
  and `5x5` morphological close, then separated complete components into upper/lower rows and five
  horizontal groups. No printed pixels, form lines, GPT candidates, or semantic labels were sent.
- **Result**: Cleanup fixed segmentation: the full blue-only crop collapsed three overlapping HASH
  detections to one. It did not make transcription reliable. Full clean variants returned
  `קנד צץ רמתא`, `קד ֶצ רמתא`, or `קד ֶצ רמא`; component rows remained incorrect. Correctly isolated
  groups returned `רבן`, `ב`, `רמת על`, `אצף`, and `הנצהר`. The numeric `23/4` group becoming `אצף`
  confirms HASH is unsuitable for digits here.
- **Limit**: This region has no verified human transcription, so the test proves a segmentation
  improvement but cannot establish an accuracy improvement. Visual inspection confirms the masks
  preserve complete connected strokes and contain no cross-row leakage.
- **Artifacts**: Ignored folder
  `analysis_results/חיוב חשבון2.pdf__2026-07-15_10-07-50/region_04_ivrit_noise_test/` contains 13
  image variants, both contact sheets, raw `ocr_results.json`, and `SUMMARY.md`.
- **Decision**: **DO NOT ADOPT** as a global HASH input or authoritative reading. Retain the existing
  narrow duplicate-line recovery. A future component-level trial must exclude numeric groups and
  require agreement with Core4 or another independent visual source.

### EXP-20260715-23: Kahlon stacked-form detection geometry

- **Status**: ADOPTED / DETECTION GEOMETRY
- **Failure**: Run `דוח_כחלון.pdf__2026-07-15_11-41-33` produced nine regions, including empty
  header/table boxes, overlapping crops, printed-only body text, and lower-form labels mapped onto
  the first declaration. Six final OCR regions were empty. The original detector map did not cover
  the actual stacked declaration fields reliably.
- **Root cause**: When the coarse page scan saw only the header, one `LOWER 80%` recovery image still
  contained the table plus both declarations. The model stopped early and its candidates were
  appended to the already incomplete coarse scan. Compact `date`/`name`/ID boxes were then expanded
  through neighboring black print by generic ink chaining.
- **Change**: Detection now sends aligned original and pen-stroke-isolated support views. A coarse
  scan that truly lacks lower-page coverage triggers five overlapping vertical segments
  (`0-30%`, `24-48%`, `42-66%`, `60-82%`, `76-100%`). Complete tiled coverage replaces only that
  incomplete coarse result; partial tile failures retain coarse candidates, and tiles can recover
  from a coarse-call timeout. A broad low-count trigger was tested and removed because it increased
  model variance on otherwise complete pages. Structured labels keep local model geometry instead
  of generic ink expansion.
- **Monochrome cleanup**: Compact black-pen candidates below 0.6% thick-pen density are removed.
  Same-kind overlapping tile boxes retain the crop with less stroke contact at its borders. A weak
  compact candidate below 23% thick-stroke retention can move only to a nearby band with at least
  30% retention, a 10-point improvement, and at least two glyph-like components; this prevents a
  signature stroke from winning a date relocation.
- **Peer recovery**: On monochrome declaration rows, a compact detected name/ID/entry anchors the
  row. Large thick-stroke component groups recover an omitted peer name or ID on that same line.
  A gated post-dedup pass restores other uncovered groups with at least three large components;
  the two-component signature remains excluded. It is disabled on colored pages and does not
  activate on Elad's ruled form rows. Generic number fields are excluded from duplicate removal
  after replay caught an `אישור מסירה` building/apartment regression.
- **Frozen replay**: The saved section candidates reduced from 11 to eight text fields. The empty
  table cell and two clipped duplicates were removed. The misplaced first-declaration date moved
  from printed text at `[1864, 1708, 389, 130]` to the complete handwritten `10/7/26` crop at
  `[1867, 1987, 402, 99]`. Artifacts are in the ignored Kahlon run folder as
  `detection_sectioned_probe.*`, `detection_cleaned_probe.png`, and
  `detection_cleaned_date.png`.
- **Validation**: Eight focused detector regressions and the complete suite pass. Total suite:
  149 tests before the dedicated benchmark additions. A detection-only matrix against human
  geometry achieved full recall on five known-good documents: `אישור מסירה` 8/8, `דוח_אלעד` 6/6,
  `דוח_יוסי` 10/10, `חיוב חשבון1` 20/20, and `חיוב חשבון2` 13/13. Deterministic saved-map replay
  preserved exact counts and zero center drift for all five. The final source-to-detector Kahlon
  run scores 8/8 with zero extras and 0.010 average normalized center drift against its local
  detection-only golden. `detection_regression.py` makes this matrix repeatable. Final full-suite,
  compilation, editor-diagnostic, and `git diff --check` results pass; the final suite contains
  157 tests.
- **Decision**: **ADOPTED** as general stacked-form recovery and conservative monochrome compact
  geometry. Generated probe artifacts remain ignored and are not runtime inputs.

### EXP-20260715-24: Cross-document deterministic quality stack and bounded rescue

- **Status**: **ADOPTED**, with broad high-reasoning and generic numeric/date rescues **REJECTED**.
- **Question/Hypothesis**: Can independent evidence and narrow deterministic gates improve the
  human-golden corpus without exposing answers to inference, increasing normal model calls, or
  regressing known-good fields?
- **Input and reference**: Six established saved runs (`אישור מסירה`, `חיוב חשבון1/2`, Elad,
  Yossi, Kahlon), plus fresh current-code runs for `חוזר מכר` and `פרוטוקול`. Human goldens were
  opened only during scoring.
- **Inference isolation**: Runtime prompts received no golden text. `quality_replay.py` completes
  detector cleanup and all postprocessing before importing golden values for measurement. The
  high-reasoning review received only crop, highlighted page context, and the current unresolved
  candidate.
- **Model/config**: GPT-5.4 medium for normal detection/OCR; GPT-5.4 high only for rescue
  experiments; two configured endpoints; HASH v1 local MPS sidecar as advisory evidence. Fresh
  uncovered-document runs used 10 workers and default Core4 multi-image OCR.
- **Accepted deterministic rules**: Strict eligible HASH replacement; numeric mark separation;
  numeric `S→5` homoglyph normalization; repeated-date consensus; unique one-edit CU numeric
  consensus; repeated structured-field consensus; dual-source uncertain repeated-number
  consensus; and blue/black signature geometry. CU is collected after model reconciliation and
  supplied only to deterministic rules when nearby model context is disabled.
- **Accepted runtime rescue**: A second high-reasoning reread triggers only for a non-signature
  `handwritten_text` value containing `[?]` and no resolved information. Acceptance requires
  Hebrew output and confidence `>=0.70`. The exhaustive eligible-state scan found one explicit
  uncertainty-only field; saved high reasoning recovered exact `הפועלים` at 0.72. Empty fields,
  partial words, English semantic descriptions, signatures, and lower-confidence outputs remain
  unchanged. Raw evidence is persisted as `high_uncertainty_recheck.json` and
  `high_uncertainty_evidence`.
- **Accepted geometry/output cleanup**: An exact descriptive-label compact fragment with >=45%
  containment, >=1.5 area ratio, nearby centers, and <=5% page area is removed in favor of its
  complete larger crop. Empty `no_handwriting` output is excluded only when every blind vision
  read is empty; the record remains in `rejected_regions`. The detection scorer now permits a
  tight crop >=80% contained in a broad human box while covering >=5% of that target.
- **Established six-run result**: Exact text 19/66 -> 28/66 (28.8% -> 42.4%); strict normalized
  similarity 0.658 -> 0.701; classification 5/7 -> 7/7. No additional model calls are required by
  this deterministic stack.
- **Eight-document result**: Exact text 20/70 -> 31/70; similarity 0.646 -> 0.704;
  classification 5/7 -> 7/7. `חוזר מכר` replays from 1/3 to 3/3 exact and 0.611 to 1.000
  similarity: duplicate top-note geometry collapses, and independent CU corrects phone
  `058329→058328`. `פרוטוקול` emits one target region after dropping an empty seam false positive,
  but its date remains 0/1 exact.
- **Rejected high-cost experiments**: Broad high-reasoning contextual rereads produced only one
  clear exact gain and several confident semantic regressions (`בי"ב` became an English mark
  description; bank-name crops became signatures; names and dates remained wrong). Date-specific
  protocol rereads correctly inferred field type but disagreed across `9/12`, `24/4/14`,
  `29/04/1[?]`, and other values. Blue-only whole-date, RTL-group, and mechanically segmented
  three-group audits also disagreed. A candidate-free numeric audit recovered the final phone
  glyph as `8`, but failed badly on the only other eligible numeric field; no generic glyph rescue
  was integrated.
- **Artifacts**: `analysis_results/high_context_review__2026-07-15/results.json`, fresh ignored
  `חוזר מכר...__quality-current__2026-07-15/` and
  `פרוטוקול...__quality-current__2026-07-15/` folders, plus per-region date/numeric review JSON.
- **Validation**: Permanent trigger, acceptance, CU-flow, output-filter, detector-dedup, scorer,
  and replay tests cover every accepted gate. `quality_replay.py` reproduces the six- and
  eight-document totals without model calls. Final suite: 182 tests; compilation, editor
  diagnostics, and `git diff --check` are clean.
- **Decision**: Keep deterministic, independently corroborated changes and the single explicit
  uncertainty-only fallback. Preserve the protocol date as unresolved; do not force a plausible
  date or deploy broad high reasoning.

### EXP-20260715-25: Candidate-blind Hebrew-versus-numeric script conflict

- **Status**: **ADOPTED / REVIEW-GATED**.
- **Failure**: On `חיוב חשבון2` region 13, human text `דנית ראובן` repeatedly became numeric
  variants (`12/2/13`, `1242 13`, `2142 13`). Core4 first classified the cursive Hebrew strokes as
  digits. The HASH result contained nine Hebrew letters but was rejected as
  `structured_or_numeric_region`; the ordinary contextual reread then received the numeric
  candidate plus a preservation instruction and repeated it.
- **Root cause**: Circular script anchoring, not HASH replacement. Disabling HASH would remove the
  only independent evidence that the region may be Hebrew while leaving the original Core4
  numeric mistake unchanged.
- **Change**: Before ordinary context, numeric-only Core4 output plus substantial Hebrew HASH
  evidence can trigger two high-reasoning candidate-blind audits. Each uses one separate endpoint
  and receives original, blue-only, local-context, and full-page images but no OCR strings. A
  Hebrew `name/address/free_text` value is emitted only when both audits exactly agree and both
  confidences are >=0.55. Hebrew-text disagreement, Hebrew-vs-numeric disagreement, one Hebrew
  result plus a timeout, or two endpoint timeouts produce `[?]`, `needs_review`, and preserved raw
  audit diagnostics; false digits remain only as an alternative. Numeric-only audit evidence may
  retain the numeric result.
- **Safety gate**: Trigger only for strong name-slot labels, or generic colored handwriting with
  at least three HASH line fragments; exclude dates, phone/email/account/reference/ID, signatures,
  marks, amounts, totals, numbers, and codes. Eight-document saved-run replay triggered exactly
  one of 77 emitted regions: this failing field.
- **Candidate-blind probes**: Without golden text or prior OCR candidates, separate audits
  classified the strokes as Hebrew `name` but disagreed on the actual name (`ב[?]לין` versus
  `יובל אלון`). A later endpoint-isolated probe encountered two timeouts. Both conditions now
  resolve to `[?]` plus review rather than either false digits or a plausible unsupported name.
- **Decision**: Keep HASH enabled as the conflict signal. Prefer honest uncertain Hebrew plus
  review over a confidently wrong numeric transcription.

## 2026-08-27

### EXP-20260827-01: Kraken PP-OCRv6 small and medium

- **Status**: REJECTED for runtime use; benchmark retained.
- **Question/Hypothesis**: The multilingual PP-OCRv6 Kraken checkpoints might provide useful
  deterministic Hebrew handwriting recognition. Zenodo reports in-domain Hebrew CER of 6.50% for
  small (`21788405`) and 5.06% for medium (`21788410`).
- **Model/config**: Kraken 7.1 on CPU, native text output, RTL base direction, direct whole-line
  recognition. Small is 3.24M parameters; medium is 15.92M. Downloaded checkpoints matched the
  published MD5 values. MPS was unusable because Kraken mixed MPS and CPU tensors.
- **Cross-region result**: On 14 original crops from
  `analysis_results/דוח_יוסי.pdf__2026-08-25_09-14-19/`, small scored 2/14 normalized exact and
  0.3262 mean similarity in 63.0 seconds. Medium scored the same 2/14 exact and 0.3536 similarity
  in 64.9 seconds. Both frequently returned short fragments, empty text, or the wrong script.
- **Requested crop result**: Kraken's baseline segmenter found no lines in
  `analysis_results/01_original_crop.png__2026-07-14_10-35-43/page_1_regions.png`. After splitting
  its clean line-removed source into three fixed line bands, small scored 0.4634 mean similarity
  and medium scored 0.4792. Medium predictions were `ביה הבועלים`, `סני4 כארה ה'רהוול`, and
  `הסית הר אותר` against `בנק הפועלים`, `סניף פתח תקווה`, and `הרצל אחד`; none was exact.
- **Clean word counterexample**: On
  `analysis_results/01_original_crop.png__2026-07-14_21-29-03/region_01_handwritten_Hebrew_word_/01_original_crop.png`,
  the human golden is `רחלי רחלה`. Small returned `דהלי רחלה` (0.7500 similarity), while medium
  returned `דח8 (חה` (0.4615). Medium is therefore not consistently better despite its aggregate
  advantage.
- **Artifacts**: `kraken_benchmark.py`; ignored model files under `.models/`; ignored reports and
  line crops under `analysis_results/`.
- **Decision and next step**: Medium is slightly better than small but remains unusable on these
  modern blue-ink forms. Do not add either base checkpoint to production or advisory OCR. A future
  trial requires accurate line segmentation and Hebrew-specific fine-tuning on representative
  contemporary forms.

### EXP-20260827-02: Gated Kraken-small pipeline ablation

- **Status**: AVAILABLE / UNMEASURED.
- **Question/Hypothesis**: Although PP-OCRv6-small is weak standalone, a correlated reading may
  occasionally repair one ambiguous Hebrew letter when GPT Core4 and visual reconciliation drift.
- **Change**: Added the opt-in UI strategy **Experimental standard + Kraken small advisory**.
  The control strategy leaves Kraken disabled. Treatment runs execute the local CPU checkpoint in
  parallel with GPT, persist `kraken_ocr.json`, and admit only nonnumeric single-line Hebrew with
  at least 50% character agreement and 60% length overlap with Core4. Duplicate and unrelated
  readings remain diagnostic only. Kraken is excluded from deterministic voting and semantics;
  eligible text is only a weak candidate in image-grounded visual adjudication.
- **Safety/operations**: Missing executable, model, timeout, and recognition failures are nonfatal.
  One Kraken worker prevents multiple PyTorch imports from spiking memory. The existing GPT model,
  prompts, and fast-profile settings are identical between control and treatment.
- **Verification**: Adapter, gate, strategy environment, persistence, and reconciliation-boundary
  tests pass; the complete app module passes 158 tests.
- **Decision and next step**: Run paired control/treatment analyses on identical selected pages and
  score both raw `handwriting_result.json` files against the same human golden. Do not promote this
  strategy unless exact text or mean similarity improves across multiple document types without
  classification regressions.

### EXP-20260827-04: Switch pipeline ablation to Kraken medium

- **Status**: AVAILABLE / UNMEASURED.
- **Change**: New treatment runs now use the verified PP-OCRv6-medium checkpoint from Zenodo
  `21788410` (15.92M parameters, MD5 `e3411a453ce3b9e9efae3f8631d85762`) and source identifier
  `kraken_ppocrv6_medium`. Historical small-run metadata remains readable.
- **Reason**: Medium scored slightly higher than small on the 14-region standalone benchmark
  (0.3536 versus 0.3262 mean similarity), although it regressed on one clean word crop. The same
  conservative gate and visual-only advisory role remain unchanged.
- **Decision and next step**: Repeat the paired control/treatment run. Do not infer improvement from
  the model-size change until the medium treatment is scored against the same human golden.

### EXP-20260827-05: MiDRASH Geniza 01 medieval HTR model

- **Status**: REJECTED for modern forms.
- **Model**: `MiDRASH_Gen_01.mlmodel` from Zenodo `10.5281/zenodo.17931017`, checksum
  `f42d75dd8b5fa99d136ac6ddf6606172`. Kraken 7.1 loaded the 23 MB checkpoint successfully.
  The newer v2 record only corrects metadata. Upstream reports 86.78% in-domain accuracy and uses
  NFKD normalization. License: CC-BY-NC-SA-4.0.
- **Domain**: Medieval Cairo Genizah documentary and literary manuscripts in Hebrew, Aramaic, and
  Judeo-Arabic, with manuscript-specific transcription conventions.
- **Result**: On the clean `רחלי רחלה` crop it returned `נתן רה` (0.3077 similarity). On the
  three-line sample it averaged 0.3638 similarity. On the same 14-region modern-form benchmark used
  for PP-OCRv6, it scored 0/14 exact and 0.0679 mean similarity in 72.2 seconds. Several outputs
  were empty or contained non-Hebrew Coptic-like graphemes.
- **Comparison**: PP-OCRv6 medium scored 2/14 exact and 0.3536 mean similarity on the 14-region
  benchmark. MiDRASH therefore regressed substantially despite its strong medieval in-domain score.
- **Decision**: Do not add MiDRASH to the app pipeline. Retain the ignored checkpoint and benchmark
  artifacts only for reproducibility; its non-commercial license and severe domain mismatch both
  preclude production adoption.

### EXP-20260827-03: Restore GPT-5.4 detection quality profile

- **Status**: ADOPTED.
- **Observed regression**: A Kraken treatment run on `אישור מסירה -דירה 8.pdf` detected only six
  regions, merging or omitting the top date/building/apartment fields and adding redaction-adjacent
  partial-stroke false positives. The known detection target count is eight.
- **Root cause**: The UI's GPT-5.4 fast profile had changed detection from the pipeline's known-good
  medium reasoning, 6,000-token settings to low reasoning and 2,500 tokens with a 60-second global
  read timeout. This affected both standard and Kraken treatment runs; Kraken itself runs only after
  region detection and did not cause the missing boxes.
- **Change**: Keep handwriting OCR at low reasoning, 4,000 tokens, and no final original rescue, but
  restore detection to medium reasoning, 6,000 tokens, and a 90-second read timeout for both
  `standard` and `standard-kraken-small`.
- **Live validation**: Detection-only rerun on the exact sample passed the human geometry gate with
  recall 8/8, zero extras, and average center distance 0.045. Artifact:
  `analysis_results/detection_regression__2026-08-27_restored-profile/`.
- **Decision**: Preserve detector quality independently from OCR speed tuning. The completed
  six-region run is invalid for the Kraken A/B and must be rerun from the upload screen.

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

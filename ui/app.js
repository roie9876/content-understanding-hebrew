const uploadView = document.querySelector("#uploadView");
const selectionView = document.querySelector("#selectionView");
const workspace = document.querySelector("#workspace");
const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const modelSelect = document.querySelector("#modelSelect");
const strategySelect = document.querySelector("#strategySelect");
const uploadError = document.querySelector("#uploadError");
const jobError = document.querySelector("#jobError");
const sourceImage = document.querySelector("#sourceImage");
const sourceCanvas = document.querySelector("#sourceCanvas");
const resultEmpty = document.querySelector("#resultEmpty");
const resultEmptyText = document.querySelector("#resultEmptyText");
const contentOutput = document.querySelector("#contentOutput");
const handwritingFrame = document.querySelector("#handwritingFrame");
const mergedDocumentFrame = document.querySelector("#mergedDocumentFrame");
const regionsImage = document.querySelector("#regionsImage");
const adminOutput = document.querySelector("#adminOutput");
const goldenOutput = document.querySelector("#goldenOutput");
const downloadPngButton = document.querySelector("#downloadPngButton");
const selectionError = document.querySelector("#selectionError");
const pageRangeInput = document.querySelector("#pageRangeInput");
const thumbnailGrid = document.querySelector("#thumbnailGrid");
const selectionPreviewImage = document.querySelector("#selectionPreviewImage");
const savedRunsList = document.querySelector("#savedRunsList");
const comparisonWorkspace = document.querySelector(".comparison-workspace");
const workspaceSplitter = document.querySelector("#workspaceSplitter");
const resultCanvas = document.querySelector("#resultCanvas");
const zoomLevel = document.querySelector("#zoomLevel");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomInButton = document.querySelector("#zoomInButton");

let activeJob = null;
let activeUpload = null;
let selectedPages = new Set();
let previewPage = 1;
let activePage = 1;
let activeView = "content";
let pollTimer = null;
let loadedContentUrl = null;
let diagnosticRequest = 0;
let goldenRequest = 0;
let processZoom = Number(localStorage.getItem("inkSightProcessZoom")) || 100;

function setProcessZoom(percent, persist = true) {
  processZoom = Math.max(50, Math.min(200, Math.round(percent / 25) * 25));
  resultCanvas.style.setProperty("--process-zoom", String(processZoom / 100));
  zoomLevel.textContent = `${processZoom}%`;
  zoomOutButton.disabled = processZoom === 50;
  zoomInButton.disabled = processZoom === 200;
  if (persist) localStorage.setItem("inkSightProcessZoom", String(processZoom));
}

zoomOutButton.addEventListener("click", () => setProcessZoom(processZoom - 25));
zoomInButton.addEventListener("click", () => setProcessZoom(processZoom + 25));
document.querySelector("#zoomResetButton").addEventListener("click", () => setProcessZoom(100));
setProcessZoom(processZoom, false);

function setSourcePanelWidth(percent, persist = true) {
  const clamped = Math.max(20, Math.min(80, percent));
  comparisonWorkspace.style.setProperty("--source-panel-width", `${clamped}%`);
  workspaceSplitter.setAttribute("aria-valuenow", String(Math.round(clamped)));
  if (persist) localStorage.setItem("inkSightSourcePanelWidth", String(clamped));
}

function resizeWorkspace(event) {
  const bounds = comparisonWorkspace.getBoundingClientRect();
  if (!bounds.width) return;
  setSourcePanelWidth(((event.clientX - bounds.left) / bounds.width) * 100);
}

let resizingWorkspace = false;
document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest("#workspaceSplitter")) return;
  event.preventDefault();
  resizingWorkspace = true;
  comparisonWorkspace.classList.add("resizing");
  resizeWorkspace(event);
});
document.addEventListener("pointermove", (event) => {
  if (resizingWorkspace) resizeWorkspace(event);
});
document.addEventListener("pointerup", () => {
  resizingWorkspace = false;
  comparisonWorkspace.classList.remove("resizing");
});
document.addEventListener("dblclick", (event) => {
  if (event.target.closest("#workspaceSplitter")) setSourcePanelWidth(50);
});
document.addEventListener("keydown", (event) => {
  if (!event.target.closest("#workspaceSplitter")) return;
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const current = Number(workspaceSplitter.getAttribute("aria-valuenow")) || 50;
  if (event.key === "Home") setSourcePanelWidth(20);
  else if (event.key === "End") setSourcePanelWidth(80);
  else setSourcePanelWidth(current + (event.key === "ArrowRight" ? 5 : -5));
});
setSourcePanelWidth(Number(localStorage.getItem("inkSightSourcePanelWidth")) || 50, false);

const stageLabels = {
  queued: "Queued",
  preparing: "Preparing",
  rendered: "Page rendered",
  content_understanding: "Content Understanding",
  detecting_handwriting: "Detecting handwriting",
  handwriting_ocr: "Handwriting OCR",
  reconciling: "Reconciling",
  merging: "Merging output",
  complete: "Complete",
  cancelling: "Stopping",
  cancelled: "Cancelled",
  failed: "Failed",
  waiting: "Waiting",
};

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function formatTime(seconds) {
  if (seconds === null || seconds === undefined) return "--:--";
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function formatRunDate(timestamp) {
  if (!timestamp) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp * 1000));
}

function setError(element, message) {
  element.textContent = message || "";
  element.hidden = !message;
}

function cleanContentUnderstanding(markdown) {
  return (markdown || "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/^\s*figures\/\S+.*$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = text;
  return element;
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = String(value ?? "");
  return element.innerHTML;
}

function parsePageSpec(spec, total) {
  const pages = new Set();
  for (const rawPart of spec.split(",")) {
    const part = rawPart.trim();
    if (!part) continue;
    if (part.includes("-")) {
      const match = part.match(/^(\d+)\s*-\s*(\d+)$/);
      if (!match) throw new Error(`Invalid page range: ${part}`);
      const start = Number(match[1]);
      const end = Number(match[2]);
      if (start > end) throw new Error(`Page range must be ascending: ${part}`);
      for (let page = start; page <= end; page += 1) pages.add(page);
    } else if (/^\d+$/.test(part)) {
      pages.add(Number(part));
    } else {
      throw new Error(`Invalid page number: ${part}`);
    }
  }
  const invalid = [...pages].find((page) => page < 1 || page > total);
  if (invalid) throw new Error(`Page ${invalid} is outside this ${total}-page document.`);
  if (!pages.size) throw new Error("Select at least one page.");
  return new Set([...pages].sort((left, right) => left - right));
}

function formatPageSpec(pages) {
  const sorted = [...pages].sort((left, right) => left - right);
  const parts = [];
  for (let index = 0; index < sorted.length; index += 1) {
    const start = sorted[index];
    let end = start;
    while (index + 1 < sorted.length && sorted[index + 1] === end + 1) {
      end = sorted[++index];
    }
    parts.push(start === end ? String(start) : `${start}-${end}`);
  }
  return parts.join(",");
}

function setPreviewPage(pageNumber) {
  const page = activeUpload?.pages.find((item) => item.page === pageNumber);
  if (!page) return;
  previewPage = pageNumber;
  selectionPreviewImage.src = page.preview_url;
  document.querySelector("#previewPageLabel").textContent = `Page ${pageNumber}`;
  thumbnailGrid.querySelectorAll(".page-thumbnail").forEach((tile) => {
    tile.classList.toggle("viewing", Number(tile.dataset.page) === pageNumber);
  });
}

function renderPageSelection() {
  if (!activeUpload) return;
  thumbnailGrid.replaceChildren();
  activeUpload.pages.forEach((page) => {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = `page-thumbnail ${selectedPages.has(page.page) ? "selected" : ""}`;
    tile.dataset.page = page.page;
    tile.setAttribute("aria-pressed", String(selectedPages.has(page.page)));
    tile.innerHTML = `
      <span class="thumbnail-check"><i data-lucide="check"></i></span>
      <img src="${page.preview_url}" alt="Page ${page.page}" loading="lazy" />
      <span class="thumbnail-label">Page ${page.page}</span>`;
    tile.addEventListener("click", () => {
      if (selectedPages.has(page.page)) selectedPages.delete(page.page);
      else selectedPages.add(page.page);
      setPreviewPage(page.page);
      pageRangeInput.value = formatPageSpec(selectedPages);
      setError(selectionError, "");
      renderPageSelection();
    });
    thumbnailGrid.append(tile);
  });
  document.querySelector("#selectionCount").textContent = `${selectedPages.size} selected`;
  document.querySelector("#analyzeSelectedButton").disabled = selectedPages.size === 0;
  setPreviewPage(previewPage);
  refreshIcons();
}

function addAdminStat(container, label, value) {
  const stat = createElement("div", "admin-stat");
  stat.append(createElement("span", "", label), createElement("strong", "", value));
  container.append(stat);
}

function regionDetails(region, candidateCount, countLabel = "evidence reads") {
  const details = createElement("details", "region-diagnostic");
  const summary = createElement("summary");
  summary.append(createElement("span", "region-index", region.region || "?"));
  const title = createElement("span", "region-title");
  title.append(
    createElement("strong", "", region.label || `Region ${region.region || ""}`),
    createElement("span", "", `Box: ${(region.box || []).join(", ")}`),
  );
  summary.append(title, createElement("span", "region-count", `${candidateCount} ${countLabel}`));
  details.append(summary);
  return details;
}

function renderDetection(diagnostics, page) {
  const regions = diagnostics.detected?.regions || diagnostics.progress?.regions || [];
  const summary = createElement("div", "admin-summary");
  addAdminStat(summary, "Detected regions", regions.length);
  addAdminStat(summary, "Page stage", stageLabels[page.stage] || page.stage);
  adminOutput.append(summary);
  const layout = createElement("div", "detection-layout");
  if (page.regions_url) {
    const image = createElement("img");
    image.src = page.regions_url;
    image.alt = "Detected handwriting regions with numbered boxes";
    layout.append(image);
  }
  const list = createElement("div", "region-diagnostic");
  const body = createElement("div", "diagnostic-body");
  regions.forEach((region) => {
    const row = createElement("div", "region-box-row", `${region.region}. ${region.label || "Handwriting region"}`);
    row.append(createElement("code", "", `[x, y, width, height] = [${(region.box || []).join(", ")}]`));
    body.append(row);
  });
  if (!regions.length) body.append(createElement("div", "admin-empty", "Waiting for handwriting detection"));
  list.append(body);
  layout.append(list);
  adminOutput.append(layout);
}

function renderCandidates(diagnostics) {
  const progress = diagnostics.progress || {};
  const regions = progress.regions || [];
  const finalRegions = diagnostics.final_regions || [];
  const summary = createElement("div", "admin-summary");
  const completedDraws = Math.min(progress.completed || 0, progress.total || 0);
  addAdminStat(summary, "Completed draws", `${completedDraws} / ${progress.total || 0}`);
  addAdminStat(summary, "Failed attempts", progress.failed || 0);
  addAdminStat(summary, "Regions", regions.length);
  adminOutput.append(summary);
  adminOutput.append(createElement(
    "div",
    "admin-empty",
    "core4_multi is one model response synthesized from four image views. After reconciliation, visual and contextual reads appear below as separate evidence.",
  ));
  const list = createElement("div", "diagnostic-list");
  regions.forEach((region, index) => {
    const rawCandidates = region.candidates || [];
    const finalRegion = finalRegions[index] || {};
    const evidence = rawCandidates.map((candidate) => ({
      source: candidate.source === "core4_multi"
        ? "core4_multi · 4 views / 1 response"
        : candidate.source,
      text: candidate.text,
    }));
    if (finalRegion.pre_context_transcription) {
      evidence.push({ source: "visual adjudication", text: finalRegion.pre_context_transcription });
    }
    if (finalRegion.contextual_evidence) {
      evidence.push({
        source: "blind contextual reread",
        text: finalRegion.contextual_evidence.final_transcription,
      });
    }
    const details = regionDetails(region, evidence.length);
    if (index === 0 && evidence.length) details.open = true;
    const body = createElement("div", "diagnostic-body");
    evidence.forEach((candidate) => {
      const row = createElement("div", "candidate");
      row.append(
        createElement("div", "candidate-source", candidate.source),
        createElement("div", "candidate-text", candidate.text || "(empty reading)"),
      );
      body.append(row);
    });
    if (!evidence.length) body.append(createElement("div", "admin-empty", "No model response yet"));
    details.append(body);
    list.append(details);
  });
  if (!regions.length) list.append(createElement("div", "admin-empty", "Candidate collection starts after detection"));
  adminOutput.append(list);
}

function renderDecision(diagnostics) {
  const regions = diagnostics.final_regions || [];
  const summary = createElement("div", "admin-summary");
  const visualAuthority = regions.some((region) =>
    (region.notes || []).some((note) => String(note).includes("visual reconciliation"))
  );
  addAdminStat(summary, "Final decisions", regions.length);
  addAdminStat(summary, "Decision source", visualAuthority ? "Visual adjudication" : "Candidate aggregation");
  adminOutput.append(summary);
  const list = createElement("div", "diagnostic-list");
  regions.forEach((region, index) => {
    const candidates = region.per_image || [];
    const details = regionDetails(
      { ...region, region: index + 1 },
      candidates.length,
      "raw OCR responses",
    );
    const body = createElement("div", "diagnostic-body");
    body.append(
      createElement("div", "decision-selected", region.final_transcription || "(empty final transcription)"),
      createElement("div", "decision-method", `Method: ${region.vote?.method || (region.notes || []).join(" · ") || "not recorded"}`),
    );
    const alternatives = region.vote?.alternatives || region.alternative_transcriptions || [];
    if (alternatives.length) {
      const alternativeList = createElement("div", "alternative-list");
      alternatives.forEach((alternative) => alternativeList.append(createElement("span", "alternative", alternative)));
      body.append(alternativeList);
    }
    const evidence = createElement("div", "word-evidence");
    (region.words || []).forEach((word) => {
      const confidence = Math.max(0, Math.min(1, Number(word.confidence) || 0));
      const row = createElement("div", "word-row");
      row.append(createElement("span", "word-label", word.text || ""));
      const track = createElement("span", "confidence-track");
      const fill = createElement("span");
      fill.style.width = `${Math.round(confidence * 100)}%`;
      track.append(fill);
      row.append(track, createElement("span", "", `${Math.round(confidence * 100)}%`));
      if (word.reason) row.append(createElement("div", "word-reason", word.reason));
      evidence.append(row);
    });
    body.append(evidence);
    details.append(body);
    list.append(details);
  });
  if (!regions.length) list.append(createElement("div", "admin-empty", "Final vote evidence appears after reconciliation"));
  adminOutput.append(list);
}

function renderLogs(diagnostics) {
  const log = diagnostics.log || [];
  adminOutput.append(createElement("pre", "page-log", log.length ? log.join("\n") : "No page events yet."));
}

async function showAdminView(page) {
  const requestId = ++diagnosticRequest;
  adminOutput.hidden = false;
  adminOutput.replaceChildren(createElement("div", "admin-empty", "Loading page diagnostics…"));
  try {
    const response = await fetch(`${page.diagnostics_url}?t=${Date.now()}`);
    if (!response.ok) throw new Error("Diagnostics unavailable");
    const diagnostics = await response.json();
    if (requestId !== diagnosticRequest) return;
    adminOutput.replaceChildren();
    if (activeView === "detection") renderDetection(diagnostics, page);
    if (activeView === "candidates") renderCandidates(diagnostics);
    if (activeView === "decision") renderDecision(diagnostics);
    if (activeView === "logs") renderLogs(diagnostics);
  } catch (error) {
    if (requestId === diagnosticRequest) {
      adminOutput.replaceChildren(createElement("div", "admin-empty", error.message));
    }
  }
}

async function uploadDocument(file) {
  if (!file) return;
  setError(uploadError, "");
  const formData = new FormData();
  formData.append("document", file);
  uploadForm.classList.add("dragging");
  try {
    const response = await fetch("/api/uploads", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Upload failed");
    activeUpload = payload;
    selectedPages = new Set(payload.pages.map((page) => page.page));
    previewPage = 1;
    pageRangeInput.value = formatPageSpec(selectedPages);
    document.querySelector("#selectionFileName").textContent = payload.file_name;
    document.querySelector("#selectionPageCount").textContent = `${payload.page_count} ${payload.page_count === 1 ? "page" : "pages"}`;
    document.querySelector("#thumbnailPageCount").textContent = payload.page_count;
    uploadView.hidden = true;
    selectionView.hidden = false;
    renderPageSelection();
  } catch (error) {
    setError(uploadError, error.message);
  } finally {
    uploadForm.classList.remove("dragging");
  }
}

async function openSavedRun(runId) {
  setError(uploadError, "");
  try {
    const response = await fetch(`/api/jobs/${runId}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Saved analysis is unavailable");
    clearTimeout(pollTimer);
    activeJob = payload;
    activePage = payload.selected_pages[0];
    activeView = "handwriting";
    loadedContentUrl = null;
    document.querySelectorAll(".stage-tab").forEach((tab) => {
      const selected = tab.dataset.view === activeView;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-selected", String(selected));
    });
    uploadView.hidden = true;
    selectionView.hidden = true;
    workspace.hidden = false;
    renderJob();
  } catch (error) {
    setError(uploadError, error.message);
  }
}

async function loadSavedRuns() {
  savedRunsList.replaceChildren(createElement("div", "saved-runs-empty", "Loading saved analyses…"));
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error("Could not load saved analyses");
    const runs = await response.json();
    savedRunsList.replaceChildren();
    runs.forEach((run) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "saved-run";
      button.innerHTML = `
        <span class="saved-run-icon"><i data-lucide="history"></i></span>
        <span class="saved-run-copy">
          <strong>${escapeHtml(run.file_name)}</strong>
          <span>${escapeHtml(formatRunDate(run.created_at))} · ${run.selected_page_count} ${run.selected_page_count === 1 ? "page" : "pages"} · ${escapeHtml(run.model_label)}${run.ocr_strategy !== "standard" ? ` · ${escapeHtml(run.ocr_strategy_label)}` : ""}</span>
          <code>${escapeHtml(run.run_name)}</code>
        </span>
        <i class="saved-run-arrow" data-lucide="chevron-right"></i>`;
      button.addEventListener("click", () => openSavedRun(run.id));
      savedRunsList.append(button);
    });
    if (!runs.length) {
      savedRunsList.append(createElement("div", "saved-runs-empty", "Completed analyses will appear here."));
    }
    refreshIcons();
  } catch (error) {
    savedRunsList.replaceChildren(createElement("div", "saved-runs-empty", error.message));
  }
}

async function analyzeSelectedPages() {
  if (!activeUpload || !selectedPages.size) return;
  setError(selectionError, "");
  const formData = new FormData();
  formData.append("upload_id", activeUpload.id);
  formData.append("model_profile", modelSelect.value);
  formData.append("ocr_strategy", strategySelect.value);
  formData.append("pages", formatPageSpec(selectedPages));
  const button = document.querySelector("#analyzeSelectedButton");
  button.disabled = true;
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not start analysis");
    activeJob = payload;
    activePage = payload.selected_pages[0];
    selectionView.hidden = true;
    workspace.hidden = false;
    renderJob();
    schedulePoll();
  } catch (error) {
    setError(selectionError, error.message);
    button.disabled = false;
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (!activeJob || ["complete", "failed", "cancelled"].includes(activeJob.status)) return;
  pollTimer = setTimeout(pollJob, 1000);
}

async function pollJob() {
  if (!activeJob) return;
  try {
    const response = await fetch(`/api/jobs/${activeJob.id}`);
    if (response.status === 404) {
      clearTimeout(pollTimer);
      activeJob = null;
      activeUpload = null;
      selectedPages.clear();
      fileInput.value = "";
      workspace.hidden = true;
      selectionView.hidden = true;
      uploadView.hidden = false;
      await loadSavedRuns();
      setError(uploadError, "The server restarted and the active job is no longer available. Start a new analysis.");
      return;
    }
    if (!response.ok) throw new Error("Could not refresh processing status");
    activeJob = await response.json();
    renderJob();
  } catch (error) {
    setError(jobError, error.message);
  } finally {
    schedulePoll();
  }
}

function renderPageList() {
  const pageList = document.querySelector("#pageList");
  pageList.replaceChildren();
  activeJob.pages.forEach((page) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `page-item ${page.stage === "complete" ? "complete" : ""} ${page.page === activePage ? "active" : ""}`;
    button.innerHTML = `
      <span class="page-number">${page.stage === "complete" ? '<i data-lucide="check"></i>' : page.page}</span>
      <span class="page-info"><strong>Page ${page.page}</strong><span>${stageLabels[page.stage] || page.stage}</span></span>`;
    button.addEventListener("click", () => {
      activePage = page.page;
      loadedContentUrl = null;
      renderJob();
    });
    pageList.append(button);
  });
}

async function showResultView(page) {
  resultEmpty.hidden = true;
  contentOutput.hidden = true;
  handwritingFrame.hidden = true;
  mergedDocumentFrame.hidden = true;
  regionsImage.hidden = true;
  adminOutput.hidden = true;
  goldenOutput.hidden = true;
  downloadPngButton.hidden = !(activeView === "merged" && page.stage === "complete" && page.merged_png_url);

  if (activeView === "content" && page.content_url) {
    contentOutput.hidden = false;
    if (loadedContentUrl !== page.content_url) {
      contentOutput.textContent = "Loading content…";
      try {
        const response = await fetch(page.content_url);
        contentOutput.textContent = response.ok
          ? cleanContentUnderstanding(await response.text())
          : "Content output is unavailable.";
        loadedContentUrl = page.content_url;
      } catch {
        contentOutput.textContent = "Content output is unavailable.";
      }
    }
  } else if (activeView === "handwriting" && page.handwriting_url) {
    handwritingFrame.hidden = false;
    if (handwritingFrame.src !== new URL(page.handwriting_url, location.href).href) handwritingFrame.src = page.handwriting_url;
  } else if (activeView === "merged" && page.merged_document_url && page.stage === "complete") {
    mergedDocumentFrame.hidden = false;
    const mergedUrl = new URL(page.merged_document_url, location.href).href;
    if (mergedDocumentFrame.src !== mergedUrl) mergedDocumentFrame.src = page.merged_document_url;
  } else if (activeView === "golden" && page.golden_comparison_url && page.stage === "complete") {
    showGoldenComparison(page);
  } else if (["detection", "candidates", "decision", "logs"].includes(activeView)) {
    showAdminView(page);
  } else {
    if (activeView === "merged") mergedDocumentFrame.src = "about:blank";
    resultEmpty.hidden = false;
    resultEmptyText.textContent = activeView === "content"
      ? "Waiting for Content Understanding"
      : activeView === "handwriting"
        ? "Waiting for handwriting analysis"
        : activeView === "merged"
          ? "Waiting for merged document"
          : activeView === "golden"
            ? "Waiting for completed results"
        : "Waiting for analysis data";
  }
}

async function showGoldenComparison(page) {
  const requestId = ++goldenRequest;
  goldenOutput.hidden = false;
  goldenOutput.replaceChildren(createElement("div", "admin-empty", "Loading human comparison…"));
  try {
    const response = await fetch(`${page.golden_comparison_url}?t=${Date.now()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Golden comparison is unavailable");
    if (requestId !== goldenRequest) return;
    if (!payload.available) {
      goldenOutput.replaceChildren(createElement("div", "admin-empty", payload.reason || "No human golden is available for this page."));
      return;
    }

    const header = document.createElement("div");
    header.className = "golden-heading";
    header.append(
      createElement("div", "panel-kicker", "HUMAN VERIFIED"),
      createElement("h3", "", "Result compared with golden"),
      createElement("p", "", `${payload.matched_regions} human-verified regions aligned with ${payload.current_regions} detected regions.`),
    );

    const tableWrap = document.createElement("div");
    tableWrap.className = "golden-table-wrap";
    const table = document.createElement("table");
    table.className = "golden-table";
    table.innerHTML = "<thead><tr><th>Field</th><th>Human golden</th><th>OCR result</th><th>Similarity</th></tr></thead>";
    const body = document.createElement("tbody");
    payload.rows.forEach((row) => {
      const tr = document.createElement("tr");
      const score = Math.round((row.ratio || 0) * 100);
      const scoreClass = score >= 90 ? "match-high" : score >= 70 ? "match-mid" : "match-low";
      tr.innerHTML = `
        <td class="golden-field">${escapeHtml(row.golden_label || "Region")}</td>
        <td class="golden-text" dir="rtl">${escapeHtml(row.golden || "(non-text)")}</td>
        <td class="golden-text" dir="rtl">${escapeHtml(row.current || "(empty)")}</td>
        <td><span class="match-score ${scoreClass}">${score}%</span></td>`;
      body.append(tr);
    });
    table.append(body);
    tableWrap.append(table);

    const summary = document.createElement("div");
    summary.className = "golden-score-summary";
    summary.innerHTML = `
      <div class="golden-score-copy">
        <span>Overall similarity to human results</span>
        <strong>${payload.success_percent}%</strong>
      </div>
      <div class="golden-progress" role="progressbar" aria-label="Overall golden similarity" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${payload.success_percent}">
        <span style="width:${Math.max(0, Math.min(100, payload.success_percent))}%"></span>
      </div>
      <div class="golden-secondary-scores">
        <span>Exact text <b>${payload.exact_percent}%</b></span>
        <span>Classification <b>${payload.classification_percent}%</b></span>
      </div>`;
    goldenOutput.replaceChildren(header, tableWrap, summary);
  } catch (error) {
    if (requestId === goldenRequest) {
      goldenOutput.replaceChildren(createElement("div", "admin-empty", error.message));
    }
  }
}

function renderJob() {
  if (!activeJob) return;
  const page = activeJob.pages.find((item) => item.page === activePage) || activeJob.pages[0];
  document.querySelector("#fileName").textContent = activeJob.file_name;
  document.querySelector("#statusMessage").textContent = activeJob.message;
  const liveParts = [];
  if (page.process_alive) liveParts.push("Process active");
  if (page.draws_total) {
    liveParts.push(`${Math.min(page.draws_completed, page.draws_total)} / ${page.draws_total} draws complete`);
  }
  if (page.draws_failed) {
    const allDrawsComplete = page.draws_total && page.draws_completed >= page.draws_total;
    liveParts.push(allDrawsComplete
      ? `${page.draws_failed} transient timeouts recovered`
      : `${page.draws_failed} attempts timed out; retrying`);
  }
  if (page.last_activity_seconds !== null && page.last_activity_seconds !== undefined) {
    liveParts.push(`last event ${formatTime(page.last_activity_seconds)} ago`);
  }
  document.querySelector("#statusDetail").textContent = liveParts.join(" · ");
  document.querySelector("#pagesMetric").textContent = `${activeJob.processed_pages} / ${activeJob.selected_page_count}`;
  const modelLabel = activeJob.model_label || activeJob.model_profile;
  document.querySelector("#modelMetric").textContent = activeJob.ocr_strategy !== "standard"
    ? `${modelLabel} · ${activeJob.ocr_strategy_label}`
    : modelLabel;
  document.querySelector("#stageMetric").textContent = stageLabels[activeJob.stage] || activeJob.stage;
  document.querySelector("#tokensMetric").textContent = formatNumber(activeJob.tokens);
  document.querySelector("#callsMetric").textContent = formatNumber(activeJob.calls);
  document.querySelector("#timeMetric").textContent = formatTime(activeJob.elapsed_seconds);
  document.querySelector("#progressBar").style.width = `${activeJob.progress}%`;
  document.querySelector("#pageCountLabel").textContent = activeJob.selected_page_count;
  document.querySelector("#pageTokens").textContent = formatNumber(page.tokens);
  document.querySelector("#pageTime").textContent = formatTime(page.elapsed_seconds);
  document.querySelector("#cancelButton").hidden = !["queued", "running", "cancelling"].includes(activeJob.status);
  setError(jobError, activeJob.error);

  renderPageList();
  if (page.original_url) {
    sourceImage.hidden = false;
    sourceImage.src = page.original_url;
    sourceCanvas.querySelector(".canvas-empty").hidden = true;
  } else {
    sourceImage.hidden = true;
    sourceCanvas.querySelector(".canvas-empty").hidden = false;
  }
  showResultView(page);
  refreshIcons();
}

document.querySelector("#browseButton").addEventListener("click", () => fileInput.click());
document.querySelector("#refreshRunsButton").addEventListener("click", loadSavedRuns);
fileInput.addEventListener("change", () => uploadDocument(fileInput.files[0]));
uploadForm.addEventListener("submit", (event) => event.preventDefault());
["dragenter", "dragover"].forEach((type) => uploadForm.addEventListener(type, (event) => {
  event.preventDefault();
  uploadForm.classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => uploadForm.addEventListener(type, (event) => {
  event.preventDefault();
  uploadForm.classList.remove("dragging");
}));
uploadForm.addEventListener("drop", (event) => uploadDocument(event.dataTransfer.files[0]));

document.querySelector("#applyRangeButton").addEventListener("click", () => {
  try {
    selectedPages = parsePageSpec(pageRangeInput.value, activeUpload.page_count);
    previewPage = [...selectedPages][0];
    setError(selectionError, "");
    renderPageSelection();
  } catch (error) {
    setError(selectionError, error.message);
  }
});
pageRangeInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    document.querySelector("#applyRangeButton").click();
  }
});
document.querySelector("#selectAllButton").addEventListener("click", () => {
  selectedPages = new Set(activeUpload.pages.map((page) => page.page));
  pageRangeInput.value = formatPageSpec(selectedPages);
  setError(selectionError, "");
  renderPageSelection();
});
document.querySelector("#clearSelectionButton").addEventListener("click", () => {
  selectedPages.clear();
  pageRangeInput.value = "";
  renderPageSelection();
});
document.querySelector("#analyzeSelectedButton").addEventListener("click", analyzeSelectedPages);
const isolatedMultiviewModels = new Set(["gpt-5.4", "gpt-5.6-sol"]);
strategySelect.addEventListener("change", () => {
  if (strategySelect.value === "isolated-multiview" && !isolatedMultiviewModels.has(modelSelect.value)) {
    modelSelect.value = "gpt-5.6-sol";
  }
});
modelSelect.addEventListener("change", () => {
  if (strategySelect.value === "isolated-multiview" && !isolatedMultiviewModels.has(modelSelect.value)) {
    strategySelect.value = "standard";
  }
});
document.querySelector("#selectionBackButton").addEventListener("click", () => {
  activeUpload = null;
  selectedPages.clear();
  fileInput.value = "";
  selectionView.hidden = true;
  uploadView.hidden = false;
});
document.querySelector("#previewOpenButton").addEventListener("click", () => {
  const page = activeUpload?.pages.find((item) => item.page === previewPage);
  if (page) window.open(page.preview_url, "_blank", "noopener");
});

document.querySelectorAll(".stage-tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".stage-tab").forEach((item) => {
    const selected = item === tab;
    item.classList.toggle("active", selected);
    item.setAttribute("aria-selected", String(selected));
  });
  activeView = tab.dataset.view;
  if (activeJob) showResultView(activeJob.pages.find((item) => item.page === activePage));
}));

document.querySelector("#sourceOpenButton").addEventListener("click", () => {
  const page = activeJob?.pages.find((item) => item.page === activePage);
  if (page?.original_url) window.open(page.original_url, "_blank", "noopener");
});

downloadPngButton.addEventListener("click", () => {
  const page = activeJob?.pages.find((item) => item.page === activePage);
  if (!page?.merged_png_url) return;
  const link = document.createElement("a");
  link.href = page.merged_png_url;
  link.download = "";
  link.click();
});

document.querySelector("#cancelButton").addEventListener("click", async () => {
  if (!activeJob) return;
  await fetch(`/api/jobs/${activeJob.id}/cancel`, { method: "POST" });
  pollJob();
});

document.querySelector("#newDocumentButton").addEventListener("click", () => {
  clearTimeout(pollTimer);
  activeJob = null;
  activeUpload = null;
  selectedPages.clear();
  fileInput.value = "";
  workspace.hidden = true;
  selectionView.hidden = true;
  uploadView.hidden = false;
  loadSavedRuns();
});

document.querySelector("#themeButton").addEventListener("click", () => {
  const current = document.documentElement.dataset.theme;
  document.documentElement.dataset.theme = current === "dark" ? "light" : "dark";
});

window.addEventListener("DOMContentLoaded", () => {
  refreshIcons();
  loadSavedRuns();
});
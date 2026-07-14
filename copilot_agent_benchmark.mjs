import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { CopilotClient, defineTool } from "@github/copilot-sdk";

const configPath = process.argv[2];
if (!configPath) throw new Error("Usage: node copilot_agent_benchmark.mjs <config.json>");

const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
const outputDir = path.resolve(config.output_dir);
fs.mkdirSync(outputDir, { recursive: true });
const eventPath = path.join(outputDir, "agent_events.jsonl");
let submitted = null;

function azureBearerToken() {
  return execFileSync("az", [
    "account", "get-access-token",
    "--scope", "https://cognitiveservices.azure.com/.default",
    "--query", "accessToken",
    "--output", "tsv",
  ], { encoding: "utf8", timeout: 30000 }).trim();
}

const submitAnalysis = defineTool("submit_document_analysis", {
  description: "Submit the complete final analysis for the attached document page exactly once.",
  parameters: {
    type: "object",
    additionalProperties: false,
    required: ["regions", "merged_markdown", "notes"],
    properties: {
      regions: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["box", "label", "classification", "final_transcription", "confidence"],
          properties: {
            box: {
              type: "array",
              minItems: 4,
              maxItems: 4,
              items: { type: "integer" },
              description: "[x, y, width, height] in pixels of the attached image",
            },
            label: { type: "string" },
            classification: {
              type: "string",
              enum: ["handwritten_text", "mark_or_checkbox", "signature_non_text", "no_handwriting"],
            },
            final_transcription: { type: "string" },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
        },
      },
      merged_markdown: { type: "string" },
      notes: { type: "string" },
    },
  },
  skipPermission: true,
  defer: "never",
  handler: async (analysis) => {
    if (submitted) return { accepted: false, error: "An analysis was already submitted." };
    submitted = analysis;
    return { accepted: true, region_count: analysis.regions.length };
  },
});

const client = new CopilotClient({
  workingDirectory: process.cwd(),
  logLevel: "error",
  useLoggedInUser: false,
  env: { ...process.env, COPILOT_DISABLE_KEYTAR: "1" },
});

const startedAt = Date.now();
let session;
let finalMessage = "";
try {
  await client.start();
  session = await client.createSession({
    model: "gpt-5.4",
    reasoningEffort: config.reasoning,
    reasoningSummary: "none",
    provider: {
      type: "openai",
      wireApi: "responses",
      baseUrl: config.endpoint.replace(/\/$/, ""),
      modelId: "gpt-5.4",
      wireModel: config.deployment,
      bearerTokenProvider: async () => azureBearerToken(),
      maxPromptTokens: 114000,
      maxOutputTokens: 24000,
    },
    tools: [submitAnalysis],
    availableTools: ["submit_document_analysis"],
    infiniteSessions: { enabled: false },
    memory: { enabled: false },
    enableSessionStore: false,
    enableSkills: false,
    enableConfigDiscovery: false,
    skipCustomInstructions: true,
    systemMessage: {
      mode: "append",
      content: `You are the sole end-to-end document analysis agent for a controlled benchmark.
Inspect the attached Hebrew document page directly. Detect every genuinely handwritten region,
including handwritten digits, dates, marks, and signatures. Exclude ordinary printed text.
Transcribe only visible strokes; use [?] for uncertain characters and never invent missing text.
Return boxes in attached-image pixels. Preserve numeric left-to-right order even inside Hebrew.
Classify signatures and marks rather than forcing them into Hebrew text. Build merged_markdown as
a clean page transcription with handwriting wrapped in <span style="color:#d00">...</span>.
You must finish by calling submit_document_analysis exactly once with the complete final answer.
Do not ask questions and do not rely on expected answers, golden data, or prior runs.`,
    },
  });
  session.on((event) => {
    fs.appendFileSync(eventPath, `${JSON.stringify(event)}\n`, "utf8");
  });

  const response = await session.sendAndWait({
    prompt: `Analyze page ${config.page} of ${config.document_name} end to end. The attached image is ${config.image_width} by ${config.image_height} pixels. Submit all regions and merged Markdown when complete.`,
    attachments: [{
      type: "file",
      path: path.resolve(config.image_path),
      displayName: `${config.document_name}-page-${config.page}.png`,
    }],
  }, config.timeout_seconds * 1000);
  finalMessage = response?.data?.content || "";

  if (!submitted) {
    const retry = await session.sendAndWait(
      "You have not called submit_document_analysis. Do so now with your complete final analysis.",
      Math.min(config.timeout_seconds, 300) * 1000,
    );
    finalMessage = retry?.data?.content || finalMessage;
  }
  if (!submitted) throw new Error("Agent completed without submitting document analysis");

  const elapsedSeconds = Math.round((Date.now() - startedAt) / 100) / 10;
  const result = {
    page: config.page,
    agent_benchmark: {
      profile: config.profile,
      deployment: config.deployment,
      reasoning: config.reasoning,
      elapsed_seconds: elapsedSeconds,
    },
    regions: submitted.regions,
    elapsed_seconds: elapsedSeconds,
    elapsed: `${String(Math.floor(elapsedSeconds / 60)).padStart(2, "0")}:${String(Math.round(elapsedSeconds % 60)).padStart(2, "0")}`,
  };
  fs.writeFileSync(path.join(outputDir, "handwriting_result.json"), `${JSON.stringify(result, null, 2)}\n`);
  fs.writeFileSync(path.join(outputDir, "merged.md"), `${submitted.merged_markdown.trim()}\n`);
  fs.writeFileSync(path.join(outputDir, "agent_notes.txt"), `${submitted.notes.trim()}\n`);
  fs.writeFileSync(path.join(outputDir, "raw_response.md"), `${finalMessage.trim()}\n`);
  process.stdout.write(`${JSON.stringify({ ok: true, regions: submitted.regions.length, elapsed_seconds: elapsedSeconds })}\n`);
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  fs.writeFileSync(path.join(outputDir, "error.txt"), `${message}\n`);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
} finally {
  if (session) {
    try { await client.deleteSession(session.sessionId); } catch {
      try { await session.disconnect(); } catch {}
    }
  }
  try { await client.stop(); } catch {}
}
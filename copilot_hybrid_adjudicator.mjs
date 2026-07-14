import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { CopilotClient, defineTool } from "@github/copilot-sdk";

const configPath = process.argv[2];
if (!configPath) throw new Error("Usage: node copilot_hybrid_adjudicator.mjs <config.json>");

const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
const outputDir = path.resolve(config.output_dir);
const eventPath = path.join(outputDir, "adjudicator_events.jsonl");
let submission = null;

function azureBearerToken() {
  return execFileSync("az", [
    "account", "get-access-token",
    "--scope", "https://cognitiveservices.azure.com/.default",
    "--query", "accessToken",
    "--output", "tsv",
  ], { encoding: "utf8", timeout: 30000 }).trim();
}

const submit = defineTool("submit_adjudicated_regions", {
  description: "Submit one final decision for every numbered evidence crop exactly once.",
  parameters: {
    type: "object",
    additionalProperties: false,
    required: ["regions", "notes"],
    properties: {
      regions: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["region_index", "classification", "final_transcription", "confidence", "reason"],
          properties: {
            region_index: { type: "integer", minimum: 1 },
            classification: {
              type: "string",
              enum: ["handwritten_text", "mark_or_checkbox", "signature_non_text", "no_handwriting"],
            },
            final_transcription: { type: "string" },
            confidence: { type: "number", minimum: 0, maximum: 1 },
            reason: { type: "string" },
          },
        },
      },
      notes: { type: "string" },
    },
  },
  skipPermission: true,
  defer: "never",
  handler: async (value) => {
    if (submission) return { accepted: false, error: "A submission already exists." };
    const indexes = value.regions.map((region) => region.region_index);
    const expected = Array.from({ length: config.regions.length }, (_, index) => index + 1);
    if (indexes.length !== expected.length || expected.some((index) => !indexes.includes(index))) {
      return { accepted: false, error: `Submit exactly regions 1 through ${expected.length}.` };
    }
    submission = value;
    return { accepted: true, region_count: indexes.length };
  },
});

const client = new CopilotClient({
  workingDirectory: process.cwd(),
  logLevel: "error",
  useLoggedInUser: false,
  env: { ...process.env, COPILOT_DISABLE_KEYTAR: "1" },
});

let session;
const startedAt = Date.now();
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
    tools: [submit],
    availableTools: ["submit_adjudicated_regions"],
    infiniteSessions: { enabled: false },
    memory: { enabled: false },
    enableSessionStore: false,
    enableSkills: false,
    enableConfigDiscovery: false,
    skipCustomInstructions: true,
    systemMessage: {
      mode: "append",
      content: `You are the final visual adjudicator in a sealed Hebrew handwriting benchmark.
    You receive numbered source-image crops and two independent OpenAI OCR candidate readings per crop.
The image is authoritative. Select or correct text only when visible strokes support it; candidates
are evidence, not truth. Preserve numeric left-to-right order. Use [?] for unresolved characters.
Classify marks and signatures instead of inventing Hebrew. Never infer an answer from field labels
or document expectations. You have no access to reference answers and must call
submit_adjudicated_regions exactly once with every numbered region.`,
    },
  });
  session.on((event) => fs.appendFileSync(eventPath, `${JSON.stringify(event)}\n`, "utf8"));

  const candidateText = config.regions.map((region) => ({
    region_index: region.region_index,
    neutral_label: region.neutral_label,
    primary: region.primary,
    secondary: region.secondary,
  }));
  const attachments = config.regions.map((region) => ({
    type: "file",
    path: path.resolve(region.evidence_path),
    displayName: `region-${String(region.region_index).padStart(2, "0")}.png`,
  }));
  const prompt = `Adjudicate all ${config.regions.length} regions. Each attachment filename maps to
region_index. Candidate JSON follows:\n${JSON.stringify(candidateText, null, 2)}`;
  let response = await session.sendAndWait({ prompt, attachments }, config.timeout_seconds * 1000);
  if (!submission) {
    response = await session.sendAndWait(
      `Call submit_adjudicated_regions now with exactly regions 1 through ${config.regions.length}.`,
      Math.min(config.timeout_seconds, 300) * 1000,
    );
  }
  if (!submission) throw new Error("Adjudicator completed without a valid submission");

  const elapsedSeconds = Math.round((Date.now() - startedAt) / 100) / 10;
  fs.writeFileSync(path.join(outputDir, "adjudication.json"), `${JSON.stringify({
    elapsed_seconds: elapsedSeconds,
    notes: submission.notes,
    regions: submission.regions,
  }, null, 2)}\n`);
  fs.writeFileSync(
    path.join(outputDir, "adjudicator_response.md"),
    `${response?.data?.content || ""}\n`,
  );
  process.stdout.write(`${JSON.stringify({ ok: true, elapsed_seconds: elapsedSeconds })}\n`);
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  fs.writeFileSync(path.join(outputDir, "adjudicator_error.txt"), `${message}\n`);
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
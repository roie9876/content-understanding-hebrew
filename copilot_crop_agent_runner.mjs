import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { CopilotClient, defineTool } from "@github/copilot-sdk";

const configPath = process.argv[2];
if (!configPath) throw new Error("Usage: node copilot_crop_agent_runner.mjs <config.json>");
const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
const outputDir = path.resolve(config.output_dir);
fs.mkdirSync(outputDir, { recursive: true });

function bearerToken() {
  return execFileSync("az", [
    "account", "get-access-token",
    "--scope", "https://cognitiveservices.azure.com/.default",
    "--query", "accessToken",
    "--output", "tsv",
  ], { encoding: "utf8", timeout: 30000 }).trim();
}

const client = new CopilotClient({
  workingDirectory: process.cwd(),
  logLevel: "error",
  useLoggedInUser: false,
  env: { ...process.env, COPILOT_DISABLE_KEYTAR: "1" },
});

async function analyzeRegion(region) {
  const regionDir = path.join(outputDir, `region_${String(region.region_index).padStart(2, "0")}`);
  fs.mkdirSync(regionDir, { recursive: true });
  const eventPath = path.join(regionDir, "events.jsonl");
  let submission = null;
  const submit = defineTool("submit_region_reading", {
    description: "Submit the final visual reading for this one crop exactly once.",
    parameters: {
      type: "object",
      additionalProperties: false,
      required: ["classification", "final_transcription", "confidence", "notes"],
      properties: {
        classification: {
          type: "string",
          enum: ["handwritten_text", "mark_or_checkbox", "signature_non_text", "no_handwriting"],
        },
        final_transcription: { type: "string" },
        confidence: { type: "number", minimum: 0, maximum: 1 },
        notes: { type: "string" },
      },
    },
    skipPermission: true,
    defer: "never",
    handler: async (value) => {
      if (submission) return { accepted: false, error: "A reading was already submitted." };
      submission = value;
      return { accepted: true };
    },
  });

  let session;
  const startedAt = Date.now();
  try {
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
        bearerTokenProvider: async () => bearerToken(),
        maxPromptTokens: 114000,
        maxOutputTokens: 12000,
      },
      tools: [submit],
      availableTools: ["submit_region_reading"],
      infiniteSessions: { enabled: false },
      memory: { enabled: false },
      enableSessionStore: false,
      enableSkills: false,
      enableConfigDiscovery: false,
      skipCustomInstructions: true,
      systemMessage: {
        mode: "append",
        content: `You are a sealed visual OCR agent for exactly one cropped document region.
All attached images are transformations of the same pixels. The original crop is authoritative.
Read only visible handwriting. Do not infer expected values from document type or field meaning.
Preserve physical left-to-right order for digits, even in Hebrew. Use [?] only for an unresolved
character. Classify signatures and marks instead of inventing text. You have no reference answer,
prior OCR, document context, memory, or external tools. Finish by calling submit_region_reading.`,
      },
    });
    session.on((event) => fs.appendFileSync(eventPath, `${JSON.stringify(event)}\n`, "utf8"));
    const attachments = region.views.map((view) => ({
      type: "file",
      path: path.resolve(view.path),
      displayName: view.name,
    }));
    let response = await session.sendAndWait({
      prompt: `Inspect region ${region.region_index}. Compare all four views and submit one final reading.`,
      attachments,
    }, config.timeout_seconds * 1000);
    if (!submission) {
      response = await session.sendAndWait(
        "Call submit_region_reading now with your final visual decision.",
        Math.min(config.timeout_seconds, 300) * 1000,
      );
    }
    if (!submission) throw new Error("Agent ended without submitting a region reading");
    const result = {
      region: region.region_index,
      box: region.box,
      classification: submission.classification,
      final_transcription: submission.final_transcription,
      confidence: submission.confidence,
      notes: [submission.notes, `Copilot SDK ${config.deployment} ${config.reasoning} crop agent`],
      elapsed_seconds: Math.round((Date.now() - startedAt) / 100) / 10,
    };
    fs.writeFileSync(path.join(regionDir, "result.json"), `${JSON.stringify(result, null, 2)}\n`);
    fs.writeFileSync(path.join(regionDir, "response.md"), `${response?.data?.content || ""}\n`);
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.stack || error.message : String(error);
    fs.writeFileSync(path.join(regionDir, "error.txt"), `${message}\n`);
    throw error;
  } finally {
    if (session) {
      try { await client.deleteSession(session.sessionId); } catch {
        try { await session.disconnect(); } catch {}
      }
    }
  }
}

async function runPool(items, concurrency) {
  const results = new Array(items.length);
  let cursor = 0;
  async function worker() {
    while (true) {
      const index = cursor++;
      if (index >= items.length) return;
      results[index] = await analyzeRegion(items[index]);
      process.stdout.write(`${JSON.stringify({
        type: "region_complete",
        region: items[index].region_index,
        completed: results.filter(Boolean).length,
        total: items.length,
      })}\n`);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
  return results;
}

try {
  await client.start();
  const results = await runPool(config.regions, config.concurrency);
  const combined = {
    page: config.page,
    method: "copilot-sdk-per-crop-agent",
    model: config.deployment,
    reasoning: config.reasoning,
    concurrency: config.concurrency,
    golden_used_during_inference: false,
    regions: results,
  };
  fs.writeFileSync(path.join(outputDir, "handwriting_result.json"), `${JSON.stringify(combined, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ type: "complete", regions: results.length })}\n`);
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  fs.writeFileSync(path.join(outputDir, "error.txt"), `${message}\n`);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
} finally {
  try { await client.stop(); } catch {}
}
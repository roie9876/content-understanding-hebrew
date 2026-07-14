import readline from "node:readline";
import { CopilotClient } from "@github/copilot-sdk";

const githubToken = process.env.COPILOT_GITHUB_TOKEN;
if (!githubToken) {
  throw new Error("COPILOT_GITHUB_TOKEN is required");
}

const client = new CopilotClient({
  workingDirectory: process.cwd(),
  logLevel: "error",
  gitHubToken: githubToken,
  useLoggedInUser: false,
  env: { ...process.env, COPILOT_DISABLE_KEYTAR: "1" },
});

function write(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function toPromptAndAttachments(content) {
  const prompt = [];
  const attachments = [];
  for (const item of content || []) {
    if (item?.type === "input_text" && item.text) {
      prompt.push(item.text);
      continue;
    }
    if (item?.type !== "input_image" || !item.image_url) continue;
    const match = /^data:([^;,]+);base64,(.+)$/s.exec(item.image_url);
    if (!match) throw new Error("Copilot SDK backend requires base64 image data URIs");
    attachments.push({
      type: "blob",
      mimeType: match[1],
      data: match[2],
      displayName: `ocr-evidence-${attachments.length + 1}`,
    });
  }
  return { prompt: prompt.join("\n\n"), attachments };
}

async function handle(request) {
  const { id, model, content, timeoutSeconds = 180 } = request;
  let session;
  try {
    const input = toPromptAndAttachments(content);
    session = await client.createSession({
      model,
      availableTools: [],
      infiniteSessions: { enabled: false },
      memory: { enabled: false },
      enableSessionStore: false,
      enableSkills: false,
      enableConfigDiscovery: false,
      skipCustomInstructions: true,
      systemMessage: {
        content: "Act only as a document vision and Hebrew handwriting OCR engine. Inspect attached image evidence directly. Follow the requested JSON schema exactly. Do not use tools or infer text unsupported by visible strokes.",
      },
    });
    const response = await session.sendAndWait(
      { prompt: input.prompt, attachments: input.attachments },
      timeoutSeconds * 1000,
    );
    write({ id, ok: true, text: response?.data?.content || "" });
  } catch (error) {
    write({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  } finally {
    if (session) {
      try { await client.deleteSession(session.sessionId); } catch {
        try { await session.disconnect(); } catch {}
      }
    }
  }
}

await client.start();
const models = await client.listModels();
write({
  type: "ready",
  models: models.map((model) => ({
    id: model.id,
    name: model.name,
    vision: Boolean(model.capabilities?.supports?.vision),
    policy: model.policy?.state,
  })),
});

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", (line) => {
  try { void handle(JSON.parse(line)); }
  catch (error) { write({ id: null, ok: false, error: error instanceof Error ? error.message : String(error) }); }
});

async function shutdown() {
  try { await client.stop(); } finally { process.exit(0); }
}
process.on("SIGTERM", () => { void shutdown(); });
process.on("SIGINT", () => { void shutdown(); });

setInterval(() => {
  if (process.ppid !== 1) return;
  try { process.kill(-process.pid, "SIGKILL"); } catch { process.exit(1); }
}, 1000).unref();

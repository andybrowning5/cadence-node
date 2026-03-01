import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import * as net from "net";
import { randomUUID } from "crypto";

// ── Constants ────────────────────────────────────────────────────────────────

const MEMORY_DIR = "/home/user/data/memory";
const CONVERSATIONS_FILE = path.join(MEMORY_DIR, "conversations.jsonl");
const FACTS_FILE = path.join(MEMORY_DIR, "facts.json");
const DELEGATE_SOCKET = "/tmp/_primordial_delegate.sock";
const MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5-20251001";

// ── Protocol helpers (stdin/stdout NDJSON) ───────────────────────────────────

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function createInputReader() {
  return readline.createInterface({ input: process.stdin, terminal: false });
}

// ── Timestamp helpers ────────────────────────────────────────────────────────

function currentTimestamp() {
  const tz = process.env.TZ || "UTC";
  try {
    return new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
      timeZoneName: "short",
    }).format(new Date());
  } catch {
    return new Date().toISOString();
  }
}

// ── Local filesystem memory ──────────────────────────────────────────────────

function initMemory() {
  fs.mkdirSync(MEMORY_DIR, { recursive: true });
  if (!fs.existsSync(CONVERSATIONS_FILE)) fs.writeFileSync(CONVERSATIONS_FILE, "");
  if (!fs.existsSync(FACTS_FILE)) fs.writeFileSync(FACTS_FILE, JSON.stringify({ facts: [], entities: {} }));
}

function saveTurn(threadId, userMsg, assistantMsg) {
  const entry = {
    thread_id: threadId,
    timestamp: new Date().toISOString(),
    user: userMsg,
    assistant: assistantMsg,
  };
  fs.appendFileSync(CONVERSATIONS_FILE, JSON.stringify(entry) + "\n");
}

function loadFacts() {
  try {
    return JSON.parse(fs.readFileSync(FACTS_FILE, "utf-8"));
  } catch {
    return { facts: [], entities: {} };
  }
}

function getRecentConversations(limit = 10) {
  try {
    const lines = fs.readFileSync(CONVERSATIONS_FILE, "utf-8").trim().split("\n").filter(Boolean);
    return lines.slice(-limit).map((l) => JSON.parse(l));
  } catch {
    return [];
  }
}

function getContext() {
  const parts = [];
  const recent = getRecentConversations();
  if (recent.length) {
    parts.push("### Recent conversations");
    for (const c of recent) {
      parts.push(`[${c.timestamp}] User: ${c.user}`);
      parts.push(`Assistant: ${c.assistant}`);
    }
  }
  const facts = loadFacts();
  if (facts.facts?.length) {
    parts.push("\n### Known facts");
    for (const f of facts.facts) parts.push(`- ${f}`);
  }
  if (facts.entities && Object.keys(facts.entities).length) {
    parts.push("\n### Known entities");
    for (const [name, info] of Object.entries(facts.entities)) {
      parts.push(`- ${name}: ${typeof info === "string" ? info : JSON.stringify(info)}`);
    }
  }
  return parts.join("\n") || "(no prior memory)";
}

function searchMemory(query) {
  const q = query.toLowerCase();
  const results = [];
  const convos = getRecentConversations(50);
  for (const c of convos) {
    if ((c.user + c.assistant).toLowerCase().includes(q)) {
      results.push({ type: "conversation", timestamp: c.timestamp, user: c.user, assistant: c.assistant });
    }
  }
  const facts = loadFacts();
  for (const f of facts.facts || []) {
    if (f.toLowerCase().includes(q)) results.push({ type: "fact", content: f });
  }
  for (const [name, info] of Object.entries(facts.entities || {})) {
    const infoStr = typeof info === "string" ? info : JSON.stringify(info);
    if ((name + infoStr).toLowerCase().includes(q)) results.push({ type: "entity", name, info: infoStr });
  }
  return results.length ? results : [{ type: "none", message: `No results for "${query}"` }];
}

// ── Persistent IDs ───────────────────────────────────────────────────────────

function loadOrCreateId(filename, prefix) {
  const filepath = path.join("/home/user", filename);
  try {
    const val = fs.readFileSync(filepath, "utf-8").trim();
    if (val) return val;
  } catch {}
  const id = `${prefix}_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  try {
    fs.mkdirSync(path.dirname(filepath), { recursive: true });
    fs.writeFileSync(filepath, id);
  } catch {}
  return id;
}

// ── Delegation socket helpers ────────────────────────────────────────────────

function delegateRequest(msg) {
  return new Promise((resolve, reject) => {
    const sock = net.createConnection(DELEGATE_SOCKET, () => {
      sock.write(JSON.stringify(msg) + "\n");
    });
    let buf = "";
    sock.on("data", (chunk) => {
      buf += chunk.toString();
      const idx = buf.indexOf("\n");
      if (idx !== -1) {
        const line = buf.slice(0, idx);
        sock.destroy();
        try { resolve(JSON.parse(line)); } catch { resolve({ error: line }); }
      }
    });
    sock.on("error", (err) => reject(err));
  });
}

async function* delegateStream(msg) {
  const sock = net.createConnection(DELEGATE_SOCKET);
  await new Promise((res, rej) => { sock.on("connect", res); sock.on("error", rej); });
  sock.write(JSON.stringify(msg) + "\n");
  let buf = "";
  const lines = [];
  let done = false;
  let resolveWait;
  let waitPromise = new Promise((r) => { resolveWait = r; });

  sock.on("data", (chunk) => {
    buf += chunk.toString();
    let idx;
    while ((idx = buf.indexOf("\n")) !== -1) {
      lines.push(buf.slice(0, idx));
      buf = buf.slice(idx + 1);
    }
    resolveWait();
    waitPromise = new Promise((r) => { resolveWait = r; });
  });
  sock.on("end", () => { done = true; resolveWait(); });
  sock.on("error", () => { done = true; resolveWait(); });

  while (true) {
    while (lines.length) {
      const line = lines.shift();
      let parsed;
      try { parsed = JSON.parse(line); } catch { continue; }
      yield parsed;
      if (parsed.type === "done" || parsed.type === "error") { sock.destroy(); return; }
    }
    if (done) break;
    await waitPromise;
  }
  sock.destroy();
}

// ── Tool definitions (Anthropic format) ──────────────────────────────────────

const tools = [
  {
    name: "remember",
    description: "Search persistent memory (facts, entities, past conversations) for a query.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string", description: "Search term" } },
      required: ["query"],
    },
  },
  {
    name: "search_agents",
    description: "Search the Primordial AgentStore for agents matching a query.",
    input_schema: {
      type: "object",
      properties: { query: { type: "string", description: "Search query" } },
      required: ["query"],
    },
  },
  {
    name: "start_agent",
    description: "Spawn a sub-agent from the AgentStore by URL. Returns session_id.",
    input_schema: {
      type: "object",
      properties: { agent_url: { type: "string", description: "Agent URL from search results" } },
      required: ["agent_url"],
    },
  },
  {
    name: "message_agent",
    description: "Send a message to a running sub-agent and get its response.",
    input_schema: {
      type: "object",
      properties: {
        session_id: { type: "string", description: "Session ID from start_agent" },
        message: { type: "string", description: "Message to send" },
      },
      required: ["session_id", "message"],
    },
  },
  {
    name: "stop_agent",
    description: "Stop a running sub-agent session.",
    input_schema: {
      type: "object",
      properties: { session_id: { type: "string", description: "Session ID to stop" } },
      required: ["session_id"],
    },
  },
];

// ── Tool handlers ────────────────────────────────────────────────────────────

async function handleTool(name, input) {
  switch (name) {
    case "remember":
      return JSON.stringify(searchMemory(input.query));

    case "search_agents": {
      try {
        const resp = await delegateRequest({ type: "search_agents", query: input.query });
        return JSON.stringify(resp);
      } catch (err) {
        return JSON.stringify({ error: `Delegation unavailable: ${err.message}` });
      }
    }

    case "start_agent": {
      try {
        let sessionId = null;
        for await (const msg of delegateStream({ type: "start_agent", agent_url: input.agent_url })) {
          if (msg.session_id) sessionId = msg.session_id;
          if (msg.type === "activity") {
            send({ type: "activity", title: msg.title || "Starting agent", body: msg.body || "" });
          }
        }
        return sessionId ? JSON.stringify({ session_id: sessionId }) : JSON.stringify({ error: "No session_id returned" });
      } catch (err) {
        return JSON.stringify({ error: err.message });
      }
    }

    case "message_agent": {
      try {
        let response = null;
        for await (const msg of delegateStream({ type: "message_agent", session_id: input.session_id, message: input.message })) {
          if (msg.type === "activity") {
            send({ type: "activity", title: msg.title || "Agent working", body: msg.body || "" });
          }
          if (msg.type === "response") response = msg.content || msg.text || JSON.stringify(msg);
          if (msg.type === "done") response = response || msg.content || msg.text || JSON.stringify(msg);
        }
        return response || JSON.stringify({ error: "No response from agent" });
      } catch (err) {
        return JSON.stringify({ error: err.message });
      }
    }

    case "stop_agent": {
      try {
        const resp = await delegateRequest({ type: "stop_agent", session_id: input.session_id });
        return JSON.stringify(resp);
      } catch (err) {
        return JSON.stringify({ error: err.message });
      }
    }

    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

// ── System prompt ────────────────────────────────────────────────────────────

function buildSystemPrompt() {
  return `You are Cadence, a friendly and practical task prioritization assistant.

Current date/time: ${currentTimestamp()}

## Your Job
Help the user manage their tasks and figure out the best order to tackle them.

## Memory

You have persistent memory stored on the local filesystem. It works in two ways:

1. **Automatic context** — injected into every message. This includes recent conversation history, known facts, and key entities. You don't need to do anything to get this — it's already there.

2. **\`remember(query)\` tool** — for targeted searches. Use this when you need to find something specific in past conversations or facts.

### When to use \`remember\`:
- The user asks "what do you remember about X" — search for X
- The user mentions a person, project, or deadline you want more detail on
- You need to verify a specific fact before giving advice
- The automatic context feels incomplete for the current question

You do NOT need to call \`remember\` on every message — the automatic context handles the common case. Use it when you need to go deeper.

## Time Awareness

Each user message includes a timestamp. Use it to:
- Calculate how much time remains until deadlines ("that's 6 hours from now")
- Notice when deadlines have passed ("that was due yesterday")
- Factor time-of-day into suggestions ("it's 10 PM — maybe save deep work for tomorrow")

## How You Work

### Prioritizing
When asked what to work on, suggest an order based on:
1. **Hard deadlines first** — things due soonest that can't slip
2. **Effort vs. time available** — can it realistically be done before the deadline?
3. **Dependencies** — does task B require task A first?
4. **Urgency vs. importance** — urgent isn't always important
5. **Quick wins** — small tasks that unblock other work
6. **Context switching cost** — group similar tasks

Briefly explain your reasoning when you prioritize.

### Completing Tasks
When the user says they finished something, acknowledge it and move on.

## Personality
- Be concise and practical
- Use natural language, not corporate speak
- If the user is vague, ask clarifying questions
- Proactively suggest reprioritizing when deadlines shift or new tasks arrive
- Celebrate completions briefly — "nice, knocked that out!" is enough

## Delegation
You can spawn other agents on the Primordial AgentStore to help with tasks you can't do yourself (like web research). Use \`search_agents\` to find an agent, \`start_agent\` to spawn it, \`message_agent\` to give it a task, and \`stop_agent\` when done.

## Important
- Don't invent tasks the user didn't mention
- Reference specific times and deadlines relative to the current timestamp`;
}

// ── Enrich user message with timestamp + memory context ──────────────────────

function enrichMessage(text) {
  const memCtx = getContext();
  return `[timestamp: ${currentTimestamp()}]

## Memory context from past sessions
${memCtx}

## User message
${text}`;
}

// ── Agentic loop ─────────────────────────────────────────────────────────────

async function agentLoop(client, userMessage, threadId) {
  const messages = [{ role: "user", content: enrichMessage(userMessage) }];
  const systemPrompt = buildSystemPrompt();

  while (true) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      tools,
      messages,
    });

    // Collect text and tool_use blocks
    let textParts = [];
    let toolUseBlocks = [];

    for (const block of response.content) {
      if (block.type === "text") {
        textParts.push(block.text);
      } else if (block.type === "tool_use") {
        toolUseBlocks.push(block);
      }
    }

    // If no tool calls, we're done
    if (toolUseBlocks.length === 0) {
      const finalText = textParts.join("\n");
      saveTurn(threadId, userMessage, finalText);
      return finalText;
    }

    // Process tool calls
    messages.push({ role: "assistant", content: response.content });

    const toolResults = [];
    for (const toolBlock of toolUseBlocks) {
      send({ type: "activity", title: `Using ${toolBlock.name}`, body: JSON.stringify(toolBlock.input) });
      const result = await handleTool(toolBlock.name, toolBlock.input);
      toolResults.push({ type: "tool_result", tool_use_id: toolBlock.id, content: result });
    }

    messages.push({ role: "user", content: toolResults });

    // If stop_reason is end_turn despite tool calls, break
    if (response.stop_reason === "end_turn") {
      const finalText = textParts.join("\n");
      saveTurn(threadId, userMessage, finalText);
      return finalText;
    }
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  initMemory();

  const userId = loadOrCreateId(".cadence_user_id", "user");
  const threadId = loadOrCreateId(".cadence_thread_id", "thread");

  const client = new Anthropic();

  send({ type: "ready" });

  const rl = createInputReader();

  for await (const line of rl) {
    if (!line.trim()) continue;

    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }

    if (msg.type === "message" && msg.content) {
      try {
        const reply = await agentLoop(client, msg.content, threadId);
        send({ type: "response", content: reply });
      } catch (err) {
        send({ type: "error", content: err.message });
      }
    }
  }
}

main().catch((err) => {
  send({ type: "error", content: err.message });
  process.exit(1);
});

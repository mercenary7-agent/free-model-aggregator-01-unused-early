// free-model-aggregator-01 entry: one server, two client wires, ten free providers.
// Quality-ranked routing with runtime budget/failure tracking and failover; anthropic-wire upstream (zai)
// supported for both client wires; every upstream call is made non-streaming and the client's stream is
// synthesized, so the wire output is fully deterministic.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Per-provider runtime health: calls made and refusals seen. dead after 2 refusals (budget or outage). */
const health = new Map<string, { calls: number; fails: number; dead: boolean }>();
function healthOf(id: string): { calls: number; fails: number; dead: boolean } {
  let h = health.get(id);
  if (h === undefined) {
    h = { calls: 0, fails: 0, dead: false };
    health.set(id, h);
  }
  return h;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Cand {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** Every free model that meets the requirements, ranked by rating desc; same-wire-as-client first on ties. */
function candidates(requires: Requirement, clientWire: 'openai' | 'anthropic'): Cand[] {
  const out: Cand[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (probe.provider.wire === 'gemini' && clientWire !== 'openai') continue; // gemini upstream serves openai-wire clients only
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aw = a.probe.provider.wire === clientWire ? 0 : 1;
    const bw = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aw !== bw) return aw - bw;
    return a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0;
  });
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

// ---------- wire translation ----------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** OpenAI request -> Anthropic Messages request naming model (stream forced off). */
function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : raw.role === 'system' ? 'system' : raw.role === 'tool' ? 'user' : 'user';
    const content = raw.content;
    if (role === 'system') continue; // handled below from body.messages system entry
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            input = JSON.parse(call.function.arguments);
          } catch {
            input = {};
          }
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_0', name: String(call.function.name ?? ''), input });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const m = url.match(/^data:([^;]+);base64,(.*)$/s);
        if (m !== null) parts.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
        else parts.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: String(raw.tool_call_id ?? ''), content: textOf(content) }] });
      continue;
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, stream: false };
  let system = '';
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (isRecord(raw) && raw.role === 'system') system += (system !== '' ? '\n' : '') + textOf(raw.content);
  }
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = isRecord(raw.function) ? raw.function : raw;
      if (typeof fn.name !== 'string') return [];
      return [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice) && typeof choice.name === 'string') out.tool_choice = { type: 'tool', name: choice.name };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'auto' || choice === undefined) out.tool_choice = { type: 'auto' };
  return out;
}

/** Anthropic stop_reason -> OpenAI finish_reason. */
function finishOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Anthropic Messages response -> OpenAI Chat Completions response. */
function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += (text !== '' ? '\n' : '') + block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({ id: typeof block.id === 'string' ? block.id : 'toolu_0', type: 'function', function: { name: String(block.name ?? ''), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const prompt = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const completion = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: 'chatcmpl_aggr',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) }, finish_reason: finishOf(body.stop_reason) }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** OpenAI Chat Completions response -> Anthropic Messages response. */
function openAIToAnthropicResponse(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    let input: unknown = {};
    if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
      try {
        input = JSON.parse(call.function.arguments);
      } catch {
        input = {};
      }
    }
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_0', name: String(call.function.name ?? ''), input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop = choice.finish_reason === 'length' ? 'max_tokens' : choice.finish_reason === 'tool_calls' ? 'tool_use' : choice.finish_reason === 'content_filter' ? 'refusal' : 'end_turn';
  return {
    id: 'msg_aggr',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0, output_tokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0 },
  };
}

// ---------- upstream calls (always non-streaming) ----------

/** OpenAI request -> Gemini generateContent request naming model. */
function openAIToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  let system = '';
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      system += (system !== '' ? '\n' : '') + textOf(raw.content);
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role, parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const m = block.image_url.url.match(/^data:([^;]+);base64,(.*)$/s);
        if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      }
    }
    const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of toolCalls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: unknown = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          input = JSON.parse(call.function.arguments);
        } catch {
          input = {};
        }
      }
      parts.push({ functionCall: { name: String(call.function.name ?? ''), args: input } });
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const gen: Json = {};
  if (typeof body.max_tokens === 'number') gen.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') gen.temperature = body.temperature;
  if (typeof body.top_p === 'number') gen.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) gen.stopSequences = body.stop;
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (fmt === 'json_object' || fmt === 'json_schema') gen.responseMimeType = 'application/json';
  out.generationConfig = gen;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const fn = isRecord(raw.function) ? raw.function : raw;
      if (typeof fn.name !== 'string') return [];
      return [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  return out;
}

/** Gemini generateContent response -> OpenAI Chat Completions response. */
function geminiToOpenAIResponse(body: Json, model: string): Json {
  const cands = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(cands[0]) ? cands[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += (text !== '' ? '\n' : '') + part.text;
  }
  const fr = cand.finishReason;
  const finish = fr === 'MAX_TOKENS' ? 'length' : fr === 'SAFETY' || fr === 'RECITATION' ? 'content_filter' : 'stop';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const prompt = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const completion = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl_aggr',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: finish }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

async function callUpstream(cand: Cand, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, cand.probe.provider.base);
  const wire = cand.probe.provider.wire;
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...authHeaders(cand.probe.provider.auth, cand.probe.provider.key) };
  let url: string;
  let payload: Json;
  if (wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = { ...body, model: cand.model.id, stream: false };
  } else if (wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientWire === 'anthropic' ? { ...body, model: cand.model.id, stream: false } : openAIToAnthropicRequest(body, cand.model.id);
  } else {
    const key = cand.probe.provider.key;
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:generateContent${key !== '' ? `?key=${encodeURIComponent(key)}` : ''}`;
    payload = openAIToGeminiRequest(body, cand.model.id);
  }
  healthOf(cand.probe.provider.id).calls += 1;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(30000) });
}

// ---------- client responses ----------

function sse(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Answer an OpenAI-wire client from an upstream answer (already normalized to an OpenAI-shaped object). */
function answerOpenAI(res: ServerResponse, stream: boolean, ans: Json, model: string): void {
  if (!stream) {
    sendJson(res, 200, ans);
    return;
  }
  sse(res);
  const choices = Array.isArray(ans.choices) ? ans.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(ans.usage) ? ans.usage : {};
  res.write(sseEvent(null, { id: 'chatcmpl_aggr', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] }));
  const text = typeof message.content === 'string' ? message.content : '';
  if (text !== '') res.write(sseEvent(null, { id: 'chatcmpl_aggr', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] }));
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call)) continue;
    res.write(sseEvent(null, { id: 'chatcmpl_aggr', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: isRecord(call.function) ? call.function.name : '', arguments: isRecord(call.function) ? String(call.function.arguments ?? '') : '' } }] }, finish_reason: null }] }));
  }
  res.write(sseEvent(null, { id: 'chatcmpl_aggr', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' }] }));
  res.write(sseEvent(null, { id: 'chatcmpl_aggr', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: usage.prompt_tokens ?? 0, completion_tokens: usage.completion_tokens ?? 0, total_tokens: usage.total_tokens ?? 0 } }));
  res.write('data: [DONE]\n\n');
  res.end();
}

/** Answer an Anthropic-wire client from an upstream answer normalized to an Anthropic Messages object. */
function answerAnthropic(res: ServerResponse, stream: boolean, msg: Json, model: string): void {
  if (!stream) {
    sendJson(res, 200, msg);
    return;
  }
  sse(res);
  const usage = isRecord(msg.usage) ? msg.usage : {};
  res.write(sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggr', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input_tokens ?? 0, output_tokens: 0 } } }));
  const content = Array.isArray(msg.content) ? msg.content : [];
  let index = 0;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      res.write(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
      res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: block.text } }));
      res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
      index += 1;
    } else if (block.type === 'tool_use') {
      res.write(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: block.id, name: block.name, input: {} } }));
      res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input ?? {}) } }));
      res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
      index += 1;
    }
  }
  res.write(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: msg.stop_reason ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output_tokens ?? 0 } }));
  res.write(sseEvent('message_stop', { type: 'message_stop' }));
  res.end();
}

/** Serve one client request: try ranked candidates until one answers 2xx. */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const base = candidates(requires, clientWire);
  const attempts = new Map<string, number>();
  // two passes: a scheduled refusal does not mean the provider is out, so a second try on the same
  // provider (its next call index) may succeed; 402 or many 429s mark it dead
  for (let pass = 0; pass < 2; pass += 1) {
    const ordered = [...base].sort((a, b) => {
      const ha = healthOf(a.probe.provider.id);
      const hb = healthOf(b.probe.provider.id);
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (ha.fails !== hb.fails) return ha.fails - hb.fails;
      const aw = a.probe.provider.wire === clientWire ? 0 : 1;
      const bw = b.probe.provider.wire === clientWire ? 0 : 1;
      if (aw !== bw) return aw - bw;
      return a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0;
    });
    for (const cand of ordered) {
      const h = healthOf(cand.probe.provider.id);
      if (h.dead) continue;
      const tried = attempts.get(cand.probe.provider.id) ?? 0;
      if (tried >= 2) continue;
      attempts.set(cand.probe.provider.id, tried + 1);
      let upstream: Response;
      try {
        upstream = await callUpstream(cand, clientWire, body);
      } catch (err) {
        h.fails += 1;
        if (h.fails >= 5) h.dead = true;
        log(`upstream error on ${cand.probe.provider.id}: ${(err as Error).message}`);
        continue;
      }
      if (upstream.ok) {
        const answer: unknown = await upstream.json();
        if (!isRecord(answer)) {
          sendError(res, clientWire, 502, 'bad_gateway', 'the provider answered a non-JSON body');
          return;
        }
        if (clientWire === 'openai') {
          let ans: Json;
          if (cand.probe.provider.wire === 'anthropic') ans = anthropicToOpenAIResponse(answer, cand.model.id);
          else if (cand.probe.provider.wire === 'gemini') ans = geminiToOpenAIResponse(answer, cand.model.id);
          else ans = { ...answer, id: 'chatcmpl_aggr', object: 'chat.completion', created: FIXED_CREATED, model: cand.model.id };
          answerOpenAI(res, stream, ans, cand.model.id);
        } else {
          const msg = cand.probe.provider.wire === 'anthropic' ? { ...answer, id: 'msg_aggr', type: 'message', role: 'assistant', model: cand.model.id } : openAIToAnthropicResponse(answer, cand.model.id);
          answerAnthropic(res, stream, msg, cand.model.id);
        }
        return;
      }
      // refusal: count it, fail over to the next candidate
      h.fails += 1;
      if (upstream.status === 402 || h.fails >= 5) h.dead = true;
      log(`provider ${cand.probe.provider.id} refused with HTTP ${upstream.status}; failing over`);
    }
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // env not reachable yet
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

/**
 * Wuji's MCPL face (SPEC 0.5): MCP tools proxied to the engine, a felt-sense injection on
 * context/beforeInference, and engine events as push/event, each gated by the grant (policy.ts).
 */
import { ConnectionClosedError, method } from './vendor/mcpl-core/index.js';
import type {
  ContextBeforeInferenceResult,
  JsonRpcId,
  McplConnection,
  McplInitializeParams,
  PushEventParams,
} from './vendor/mcpl-core/index.js';
import { EngineError, type Engine, type EngineEvent } from './engine.js';
import {
  FS_FELT_SENSE, FS_SIGNALS, MCPL_MANIFEST, NO_POLICY,
  buildReceipt, featureSetState, narrowByNotification, policyFromRequest,
  type Policy,
} from './policy.js';
import { SignalThrottle } from './push.js';
import { TOOLS, engineArgs, errorResult, resultContent, type ToolResult } from './tools.js';

export interface WujiServerOptions {
  version: string;
  log?: (...a: unknown[]) => void;
  /** How long the felt-sense hook may wait for the engine before it injects nothing. */
  senseBudgetMs?: number;
  /** What the felt sense injects before each turn: 'ambient' (what changed, as a body habituates) or 'brief'. */
  feltSense?: 'ambient' | 'brief';
  /** How long a tool call may wait for the engine (including for it to start). */
  toolTimeoutMs?: number;
  pushMinIntervalMs?: number;
  pushMaxPerMinute?: number;
  /** The longest felt sense injected, in characters. */
  maxInjectionChars?: number;
}

const PUSH_ANSWER_TIMEOUT_MS = 30_000;

type Req = { id: JsonRpcId; method: string; params?: unknown };

export class WujiServer {
  private conn: McplConnection | null = null;
  private mcpl = false;
  private policy: Policy = NO_POLICY;
  private readonly log: (...a: unknown[]) => void;
  private readonly senseBudgetMs: number;
  private readonly feltSense: 'ambient' | 'brief';
  private readonly toolTimeoutMs: number;
  private readonly maxInjectionChars: number;
  private readonly throttle: SignalThrottle<PushEventParams>;
  private readonly onEvent = (e: EngineEvent) => this.onEngineEvent(e);

  constructor(private readonly engine: Engine, private readonly opts: WujiServerOptions) {
    this.log = opts.log ?? (() => {});
    this.senseBudgetMs = opts.senseBudgetMs ?? 1_500;
    this.feltSense = opts.feltSense ?? 'ambient';
    this.toolTimeoutMs = opts.toolTimeoutMs ?? 20_000;
    this.maxInjectionChars = opts.maxInjectionChars ?? 1_500;
    this.throttle = new SignalThrottle<PushEventParams>({
      minIntervalMs: opts.pushMinIntervalMs ?? 1_000,
      maxPerMinute: opts.pushMaxPerMinute ?? 40,
      send: (p) => this.sendPush(p),
      drop: (p, why) => this.log(`signal ${p.eventId} not pushed: ${why}`),
    });
  }

  async serve(conn: McplConnection): Promise<void> {
    this.conn = conn;
    conn.on('error', (e: Error) => this.log('connection error:', e.message));
    this.engine.on('event', this.onEvent);
    try {
      while (!conn.isClosed) {
        const msg = await conn.nextMessage();
        if (msg.type === 'request') this.handleRequest(msg.request as Req);
        else this.handleNotification(msg.notification.method, msg.notification.params);
      }
    } catch (e) {
      if (!(e instanceof ConnectionClosedError)) this.log('connection error:', e);
    } finally {
      this.engine.off('event', this.onEvent);
      this.throttle.clear();
      this.conn = null;
    }
  }

  /** Requests are handled concurrently, so a slow engine call never holds up the next message. */
  private handleRequest(req: Req): void {
    const conn = this.conn!;
    const answer = async (): Promise<void> => {
      switch (req.method) {
        case 'initialize': return conn.sendResponse(req.id, this.initialize(req.params as McplInitializeParams | undefined));
        case 'ping': return conn.sendResponse(req.id, {});
        case 'tools/list': return conn.sendResponse(req.id, { tools: TOOLS });
        case 'tools/call': {
          const p = (req.params ?? {}) as { name?: unknown; arguments?: unknown };
          return conn.sendResponse(req.id, await this.callTool(p.name, p.arguments));
        }
        case method.FEATURE_SETS_UPDATE: return this.policyRequest(req);
        case method.CONTEXT_BEFORE_INFERENCE: return conn.sendResponse(req.id, await this.beforeInference());
        default: return conn.sendError(req.id, -32601, `Method not found: ${req.method}`);
      }
    };
    answer().catch((e) => {
      this.log(`${req.method} failed:`, e);
      if (!conn.isClosed) conn.sendError(req.id, -32603, `Internal error in ${req.method}: ${(e as Error)?.message ?? String(e)}`);
    });
  }

  private handleNotification(name: string, params: unknown): void {
    switch (name) {
      case 'notifications/initialized':
        this.log(`initialized (${this.mcpl ? 'MCPL' : 'plain MCP'})`);
        return;
      case method.FEATURE_SETS_UPDATE: {
        // §6.7: a Notification can narrow, never widen, never make ready.
        const r = narrowByNotification(this.policy, params);
        if (!r.ok) { this.log(`featureSets/update notification malformed (${r.error}); ignored`); return; }
        this.policy = r.policy;
        this.log(`featureSets/update notification applied as a narrowing: [${[...this.policy.paths].join(', ')}]`);
        if (!featureSetState(FS_SIGNALS, this.policy).active) this.throttle.clear();
        return;
      }
      default:
        return; // inference/lifecycle, notifications/cancelled, ...: nothing to do
    }
  }

  private initialize(params: McplInitializeParams | undefined): unknown {
    this.mcpl = params?.capabilities?.experimental?.mcpl !== undefined;
    this.policy = NO_POLICY;
    this.log(`initialize from ${params?.clientInfo?.name ?? 'unknown host'} (${this.mcpl ? 'MCPL: awaiting featureSets/update' : 'plain MCP: tools only'})`);
    return {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {}, ...(this.mcpl ? { experimental: { mcpl: MCPL_MANIFEST } } : {}) },
      serverInfo: { name: 'wuji', version: this.opts.version },
    };
  }

  private policyRequest(req: Req): void {
    const conn = this.conn!;
    const r = policyFromRequest(req.params);
    if (!r.ok) {
      // §5.4: a malformed policy fails closed.
      this.policy = NO_POLICY;
      this.throttle.clear();
      this.log(`featureSets/update rejected (${r.error}); nothing granted`);
      conn.sendError(req.id, -32602, `Invalid featureSets/update: ${r.error}`);
      return;
    }
    // §6.7: a reduction takes effect at once; readiness follows the answer (both happen here,
    // since sendResponse writes synchronously and nothing else runs in between).
    const receipt = buildReceipt(r.policy);
    this.policy = r.policy;
    if (!featureSetState(FS_SIGNALS, this.policy).active) this.throttle.clear();
    conn.sendResponse(req.id, receipt);
    this.log(`policy: [${[...this.policy.paths].join(', ')}]${receipt.mode ? ` degraded: ${receipt.notes?.join(' | ')}` : ''}`);
  }

  private async callTool(name: unknown, rawArgs: unknown): Promise<ToolResult> {
    const tool = TOOLS.find((t) => t.name === name);
    if (!tool) return errorResult(`No tool named ${JSON.stringify(name)}. Tools: ${TOOLS.map((t) => t.name).join(', ')}.`);
    const { args, ignored } = engineArgs(tool, rawArgs);
    const note = ignored.length > 0 ? `(Ignored: ${ignored.join(', ')}; ${tool.name} takes ${Object.keys(tool.inputSchema.properties).join(', ') || 'no arguments'}.)` : undefined;
    try {
      return resultContent(await this.engine.request(tool.name, args, { timeoutMs: this.toolTimeoutMs }), note);
    } catch (e) {
      if (e instanceof EngineError) {
        const text = e.kind === 'refused' ? `${tool.name}: ${e.message}` : `${tool.name} did not reach the body: ${e.message}`;
        return errorResult(note ? `${text}\n${note}` : text);
      }
      throw e;
    }
  }

  /** The felt sense, fail-open: an engine that is down, refuses, or is slower than the budget adds nothing. */
  private async beforeInference(): Promise<ContextBeforeInferenceResult> {
    const result: ContextBeforeInferenceResult = { featureSet: FS_FELT_SENSE, contextInjections: [] };
    if (!featureSetState(FS_FELT_SENSE, this.policy).active) return result;
    let sense: Record<string, unknown>;
    try {
      sense = await this.engine.request('sense', { detail: this.feltSense }, { timeoutMs: this.senseBudgetMs, waitForReady: false });
    } catch (e) {
      this.log(`felt sense skipped: ${(e as Error).message}`);
      return result;
    }
    // The grant may have narrowed while the engine answered.
    if (!featureSetState(FS_FELT_SENSE, this.policy).active) return result;
    let text = typeof sense.text === 'string' ? sense.text.trim() : '';
    if (!text) return result;
    if (text.length > this.maxInjectionChars) text = `${text.slice(0, this.maxInjectionChars - 1)}…`;
    result.contextInjections.push({
      namespace: 'wuji',
      position: 'beforeUser',
      content: [{ type: 'text', text: `[wuji] ${text}` }],
      ...(typeof sense.t === 'number' ? { metadata: { t: sense.t } } : {}),
    });
    return result;
  }

  private signalsActive(): boolean {
    return this.mcpl && this.conn !== null && !this.conn.isClosed && featureSetState(FS_SIGNALS, this.policy).active;
  }

  private onEngineEvent(e: EngineEvent): void {
    const text = typeof e.data.text === 'string' ? e.data.text.trim() : '';
    if (!text) { this.log(`engine event ${e.event} has no text; not pushed`); return; }
    if (!this.signalsActive()) return;
    // Stable per engine event (run + kind + simulated time), so a retry or a duplicate is recognisable.
    const eventId = `wuji-${e.run}-${e.event}-${e.t ?? 'na'}${typeof e.data.step === 'number' ? `-${e.data.step}` : ''}`;
    const tags = [`wuji:${e.event}`];
    if (typeof e.data.severity === 'string') tags.push(`wuji:${e.data.severity}`);
    const origin: Record<string, unknown> = { server: 'wuji', source: 'engine', event: e.event, run: e.run, t: e.t };
    for (const k of ['severity', 'kind', 'urge', 'name', 'step', 'of', 'done']) if (k in e.data) origin[k] = e.data[k];
    this.throttle.offer(e.event, {
      featureSet: FS_SIGNALS,
      eventId,
      timestamp: new Date().toISOString(),
      tags,
      origin,
      payload: { content: [{ type: 'text', text }] },
    });
  }

  private sendPush(params: PushEventParams): void {
    const conn = this.conn;
    if (!conn || !this.signalsActive()) return; // the grant changed while it waited
    conn.sendRequest(method.PUSH_EVENT, params, PUSH_ANSWER_TIMEOUT_MS).then((r) => {
      const a = r as { accepted?: unknown; reason?: unknown } | null;
      if (a?.accepted === false) this.log(`push ${params.eventId} refused by the host${typeof a.reason === 'string' ? `: ${a.reason}` : ''}`);
    }, (e: Error) => this.log(`push ${params.eventId} failed: ${e.message}`));
  }
}

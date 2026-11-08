/**
 * The engine child process (docs/engine-protocol.md): JSON Lines over stdio, requests matched to
 * responses by id, unsolicited events forwarded, restarted with backoff when it dies, and stopped
 * (shutdown, then end of stdin, then a kill) on exit. No shell is involved anywhere.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { StringDecoder } from 'node:string_decoder';

export interface HelloInfo {
  version?: string;
  model?: string;
  muscles?: number;
  t?: number;
  [k: string]: unknown;
}

/** An engine event, with the run (one engine process) it came from. */
export interface EngineEvent {
  event: string;
  t: number | null;
  data: Record<string, unknown>;
  run: string;
}

export type EngineState = 'idle' | 'starting' | 'ready' | 'waiting-to-restart' | 'stopped';

/** Why a request did not produce a result. `refused`: the engine answered ok: false. */
export class EngineError extends Error {
  constructor(message: string, readonly kind: 'refused' | 'unavailable' | 'timeout') {
    super(message);
    this.name = 'EngineError';
  }
}

export interface EngineOptions {
  /** argv: the program, then its arguments. */
  command: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  log?: (...a: unknown[]) => void;
  /** How long a new engine has to answer hello (the first `uv run` may install packages). */
  startTimeoutMs?: number;
  /** Restart backoff: first delay, ceiling, and how long a run must last to reset the delay. */
  restartMinMs?: number;
  restartMaxMs?: number;
  stableMs?: number;
  /** How long stop() waits for the engine to exit after shutdown before it kills it. */
  stopGraceMs?: number;
}

interface Pending {
  op: string;
  resolve: (r: Record<string, unknown>) => void;
  reject: (e: EngineError) => void;
  timer: NodeJS.Timeout;
}

interface Waiter {
  resolve: () => void;
  reject: (e: EngineError) => void;
  timer: NodeJS.Timeout;
}

const isObject = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);

export class Engine extends EventEmitter {
  private readonly opts: Required<Omit<EngineOptions, 'cwd' | 'env' | 'log'>> & Pick<EngineOptions, 'cwd' | 'env'>;
  private readonly log: (...a: unknown[]) => void;
  private child: ChildProcess | null = null;
  private exited: Promise<void> = Promise.resolve();
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private waiters = new Set<Waiter>();
  private restartTimer: NodeJS.Timeout | null = null;
  private restartDelay: number;
  private startedAt = 0;
  private _state: EngineState = 'idle';
  private _run = '';
  private _hello: HelloInfo | null = null;
  private _lastError: string | null = null;
  private _starts = 0;
  private helloFailure: string | null = null;

  constructor(options: EngineOptions) {
    super();
    if (!Array.isArray(options.command) || options.command.length === 0) throw new Error('engine command is empty');
    this.opts = {
      command: options.command,
      cwd: options.cwd,
      env: options.env,
      startTimeoutMs: options.startTimeoutMs ?? 300_000,
      restartMinMs: options.restartMinMs ?? 500,
      restartMaxMs: options.restartMaxMs ?? 30_000,
      stableMs: options.stableMs ?? 60_000,
      stopGraceMs: options.stopGraceMs ?? 3_000,
    };
    this.log = options.log ?? (() => {});
    this.restartDelay = this.opts.restartMinMs;
  }

  get state(): EngineState { return this._state; }
  /** Identifies the current engine process; changes at every restart. */
  get run(): string { return this._run; }
  get hello(): HelloInfo | null { return this._hello; }
  get lastError(): string | null { return this._lastError; }
  /** How many engine processes have been started. */
  get starts(): number { return this._starts; }

  /** A sentence on why the body cannot answer right now, for tool results. */
  describeUnavailable(): string {
    const why = this._lastError ? ` Last problem: ${this._lastError}` : '';
    switch (this._state) {
      case 'starting': return `The body (engine) is still starting.${why}`;
      case 'waiting-to-restart': return `The body (engine) stopped and is restarting.${why}`;
      case 'stopped': return 'The body (engine) has been shut down.';
      default: return `The body (engine) is not running.${why}`;
    }
  }

  start(): void {
    if (this._state !== 'idle' && this._state !== 'waiting-to-restart') return;
    this.restartTimer = null;
    this._state = 'starting';
    this._run = randomBytes(4).toString('hex');
    this._hello = null;
    this.helloFailure = null;
    this._starts += 1;
    this.startedAt = Date.now();
    const [file, ...args] = this.opts.command;
    const env = { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', ...(this.opts.env ?? process.env) };
    this.log(`starting engine (run ${this._run}): ${JSON.stringify(this.opts.command)}`);
    let child: ChildProcess;
    try {
      child = spawn(file, args, { cwd: this.opts.cwd, env, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, shell: false });
    } catch (e) {
      this.onExit(`could not start the engine: ${(e as Error).message}`);
      return;
    }
    this.child = child;
    let settle!: () => void;
    this.exited = new Promise<void>((r) => { settle = r; });
    let done = false;
    const finish = (code: number | null, signal: NodeJS.Signals | null, err?: Error) => {
      if (done) return;
      done = true;
      settle();
      if (this.child === child) this.child = null;
      const what = err
        ? `could not start the engine: ${err.message}${(err as NodeJS.ErrnoException).code === 'ENOENT' ? ` (is "${file}" on PATH? WUJI_ENGINE can name another command)` : ''}`
        : `the engine exited (${signal ? `signal ${signal}` : `code ${code}`})`;
      this.onExit(this.helloFailure ? `${this.helloFailure}; ${what}` : what);
    };
    child.once('error', (err) => finish(null, null, err));
    child.once('exit', (code, signal) => finish(code, signal));
    child.stdin!.on('error', (e) => this.log(`engine stdin: ${e.message}`));
    this.readLines(child.stdout!, (line) => this.onLine(line));
    this.readLines(child.stderr!, (line) => this.log(`engine: ${line}`));

    const run = this._run;
    this.send('hello', {}, this.opts.startTimeoutMs, false).then((info) => {
      if (this._run !== run || this._state !== 'starting') return;
      this._hello = info as HelloInfo;
      this._state = 'ready';
      this._lastError = null;
      this.log(`engine ready (run ${run}): ${JSON.stringify(info)}`);
      for (const w of this.waiters) { clearTimeout(w.timer); w.resolve(); }
      this.waiters.clear();
      this.emit('ready', info);
    }, (e: EngineError) => {
      if (this._run !== run || this._state !== 'starting') return;
      this.helloFailure = `hello failed: ${e.message}`;
      this.log(`${this.helloFailure}; killing this engine`);
      try { child.kill(); } catch { /* already gone */ }
    });
  }

  /**
   * Send one request and wait for its response. By default it first waits for the engine to be
   * ready, within the same timeout; with waitForReady false an engine that is not ready fails at
   * once (the felt-sense hook must never wait).
   */
  async request(op: string, args: Record<string, unknown> = {}, opts: { timeoutMs?: number; waitForReady?: boolean } = {}): Promise<Record<string, unknown>> {
    const timeoutMs = opts.timeoutMs ?? 15_000;
    const deadline = Date.now() + timeoutMs;
    if (this._state !== 'ready') {
      if (opts.waitForReady === false || this._state === 'stopped' || this._state === 'idle') {
        throw new EngineError(this.describeUnavailable(), 'unavailable');
      }
      await this.whenReady(timeoutMs);
    }
    return this.send(op, args, Math.max(1, deadline - Date.now()), true);
  }

  private whenReady(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const w: Waiter = {
        resolve,
        reject,
        timer: setTimeout(() => {
          this.waiters.delete(w);
          reject(new EngineError(this.describeUnavailable(), 'unavailable'));
        }, timeoutMs),
      };
      this.waiters.add(w);
    });
  }

  private send(op: string, args: Record<string, unknown>, timeoutMs: number, needReady: boolean): Promise<Record<string, unknown>> {
    const child = this.child;
    if (!child || !child.stdin || child.stdin.destroyed || (needReady && this._state !== 'ready')) {
      return Promise.reject(new EngineError(this.describeUnavailable(), 'unavailable'));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new EngineError(`the engine did not answer "${op}" within ${Math.round(timeoutMs)} ms`, 'timeout'));
      }, timeoutMs);
      this.pending.set(id, { op, resolve, reject, timer });
      child.stdin!.write(JSON.stringify({ id, op, args }) + '\n');
    });
  }

  private onLine(line: string): void {
    if (!line.trim()) return;
    let msg: unknown;
    try { msg = JSON.parse(line); } catch {
      this.log(`engine wrote a line that is not JSON (ignored): ${line.slice(0, 200)}`);
      return;
    }
    if (!isObject(msg)) { this.log('engine wrote a JSON value that is not an object (ignored)'); return; }
    if (typeof msg.event === 'string') {
      const ev: EngineEvent = {
        event: msg.event,
        t: typeof msg.t === 'number' ? msg.t : null,
        data: isObject(msg.data) ? msg.data : {},
        run: this._run,
      };
      this.emit('event', ev);
      return;
    }
    if (typeof msg.id === 'number' || typeof msg.id === 'string') {
      const p = this.pending.get(Number(msg.id));
      if (!p) return; // a request that already timed out
      this.pending.delete(Number(msg.id));
      clearTimeout(p.timer);
      if (msg.ok === true) p.resolve(isObject(msg.result) ? msg.result : {});
      else p.reject(new EngineError(typeof msg.error === 'string' ? msg.error : `"${p.op}" failed`, 'refused'));
      return;
    }
    this.log(`engine message with neither id nor event (ignored): ${line.slice(0, 200)}`);
  }

  private onExit(what: string): void {
    const stopping = this._state === 'stopped';
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(new EngineError(`${what} before answering "${p.op}"`, 'unavailable'));
    }
    this.pending.clear();
    if (stopping) { this.log(`engine stopped: ${what}`); return; }
    this._lastError = what;
    const lived = Date.now() - this.startedAt;
    if (lived >= this.opts.stableMs) this.restartDelay = this.opts.restartMinMs;
    const delay = this.restartDelay;
    this.restartDelay = Math.min(this.opts.restartMaxMs, this.restartDelay * 2);
    this._state = 'waiting-to-restart';
    this._hello = null;
    this.log(`${what}; restarting in ${delay} ms`);
    this.emit('exit', what);
    this.restartTimer = setTimeout(() => this.start(), delay);
  }

  /** Stop for good: shutdown, end of stdin, and a kill if it outstays the grace period. */
  async stop(): Promise<void> {
    if (this._state === 'stopped') return this.exited;
    this._state = 'stopped';
    if (this.restartTimer) { clearTimeout(this.restartTimer); this.restartTimer = null; }
    for (const w of this.waiters) { clearTimeout(w.timer); w.reject(new EngineError(this.describeUnavailable(), 'unavailable')); }
    this.waiters.clear();
    const child = this.child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return this.exited;
    try {
      if (child.stdin && !child.stdin.destroyed) {
        child.stdin.write(JSON.stringify({ id: this.nextId++, op: 'shutdown', args: {} }) + '\n');
        child.stdin.end();
      }
    } catch { /* the pipe is gone; the kill below covers it */ }
    const timedOut = await Promise.race([
      this.exited.then(() => false),
      new Promise<boolean>((r) => setTimeout(() => r(true), this.opts.stopGraceMs).unref()),
    ]);
    if (timedOut) {
      this.log('engine did not exit after shutdown; killing it');
      try { child.kill(); } catch { /* already gone */ }
      await Promise.race([this.exited, new Promise<void>((r) => setTimeout(r, 2_000).unref())]);
    }
  }

  /** Split a stream into lines on \n (a \r before it is dropped), decoding UTF-8 across chunks. */
  private readLines(stream: NodeJS.ReadableStream, onLine: (line: string) => void): void {
    const decoder = new StringDecoder('utf8');
    let buf = '';
    stream.on('data', (chunk: Buffer) => {
      buf += decoder.write(chunk);
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '');
        buf = buf.slice(nl + 1);
        onLine(line);
      }
    });
    stream.on('end', () => {
      buf += decoder.end();
      if (buf) onLine(buf.replace(/\r$/, ''));
      buf = '';
    });
    stream.on('error', (e: Error) => this.log(`engine stream: ${e.message}`));
  }
}

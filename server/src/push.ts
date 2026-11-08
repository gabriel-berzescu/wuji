/**
 * Rate limit for body signals. The engine already keeps events rare; this is the server's own
 * bound. Per kind, at most one push per `minIntervalMs`: a signal that comes sooner waits, and a
 * newer one of the same kind replaces it (the latest state of the body is what matters). Across
 * kinds, at most `maxPerMinute`; beyond that a signal is dropped.
 */
export interface ThrottleOptions<T> {
  minIntervalMs: number;
  maxPerMinute: number;
  send: (item: T) => void;
  drop?: (item: T, why: string) => void;
  now?: () => number;
}

export class SignalThrottle<T> {
  private last = new Map<string, number>();
  private waiting = new Map<string, { item: T; timer: NodeJS.Timeout }>();
  private sent: number[] = [];
  private readonly now: () => number;

  constructor(private readonly opts: ThrottleOptions<T>) {
    this.now = opts.now ?? Date.now;
  }

  offer(kind: string, item: T): void {
    const now = this.now();
    const since = now - (this.last.get(kind) ?? -Infinity);
    if (since >= this.opts.minIntervalMs && !this.waiting.has(kind)) {
      this.emit(kind, item, now);
      return;
    }
    const held = this.waiting.get(kind);
    if (held) {
      this.opts.drop?.(held.item, `replaced by a newer ${kind} signal`);
      held.item = item;
      return;
    }
    const timer = setTimeout(() => {
      const w = this.waiting.get(kind);
      this.waiting.delete(kind);
      if (w) this.emit(kind, w.item, this.now());
    }, Math.max(0, this.opts.minIntervalMs - since));
    timer.unref?.();
    this.waiting.set(kind, { item, timer });
  }

  private emit(kind: string, item: T, now: number): void {
    this.sent = this.sent.filter((t) => now - t < 60_000);
    if (this.sent.length >= this.opts.maxPerMinute) {
      this.opts.drop?.(item, `over ${this.opts.maxPerMinute} signals a minute`);
      return;
    }
    this.sent.push(now);
    this.last.set(kind, now);
    this.opts.send(item);
  }

  /** Forget everything waiting (the connection closed, or the grant was narrowed). */
  clear(): void {
    for (const w of this.waiting.values()) clearTimeout(w.timer);
    this.waiting.clear();
  }
}

/** The practice tools: each one engine operation (docs/engine-protocol.md), its schema mirroring the protocol. */

export interface ToolDefinition {
  name: string;
  title: string;
  description: string;
  inputSchema: { type: 'object'; properties: Record<string, unknown>; required?: string[]; additionalProperties: false };
}

const seconds = (description: string) => ({ type: 'number', minimum: 0, description });
const unit = (description: string) => ({ type: 'number', minimum: 0, maximum: 1, description });

export const REGIONS = ['feet', 'calves', 'thighs', 'hips', 'lower_back', 'belly', 'shoulders', 'arms', 'hands'] as const;

export const TOOLS: ToolDefinition[] = [
  {
    name: 'sense',
    title: 'Sense the body',
    description: 'Feel the body as it is now: weight and where it falls in the feet, alignment, sway, where effort or held '
      + 'tension sits, the breath, the heart. "brief" (default) is six to ten short lines; "full" adds every joint angle '
      + 'and region. The body keeps living in real time between calls; sensing changes nothing. (When the host grants it, '
      + 'a brief felt sense also arrives on its own before each turn.)',
    inputSchema: {
      type: 'object',
      properties: { detail: { type: 'string', enum: ['brief', 'full'], default: 'brief', description: 'How much to feel.' } },
      additionalProperties: false,
    },
  },
  {
    name: 'intend',
    title: 'Intend a posture',
    description: 'Set where the body should go and how fast; it moves there smoothly on its own clock. Every field is '
      + 'optional, and a field left out keeps its current value. Example: sink a little and shift toward the left foot '
      + 'over four seconds: {"sink": 0.4, "weight": 0.6, "over": 4}.',
    inputSchema: {
      type: 'object',
      properties: {
        stance: { type: 'string', enum: ['wuji', 'horse', 'bow_left', 'bow_right'], description: 'The stance of the legs.' },
        weight: unit('Share of the weight on the LEFT foot, 0..1; 0.5 is centred.'),
        sink: unit('How deep the knees bend: 0 straight .. 1 the deepest this stance allows.'),
        arms: { type: 'string', enum: ['hang', 'embrace', 'lift_sky', 'press_down', 'draw_bow_left', 'draw_bow_right'], description: 'What the arms do.' },
        head: { type: 'string', enum: ['suspended', 'tucked', 'turned_left', 'turned_right'], description: 'The carriage of the head.' },
        tailbone: unit('How far the tailbone drops (song yao: the waist lengthens and hangs): 0 only as much as sinking brings .. 1 dropped as far as it goes.'),
        over: seconds('Seconds to arrive (minimum-jerk movement).'),
      },
      additionalProperties: false,
    },
  },
  {
    name: 'tension',
    title: 'Hold or release tension',
    description: 'Set how much extra tension is held in a region, above what the posture needs: 0 released .. 1. '
      + 'Region "all" with level 0 releases everything.',
    inputSchema: {
      type: 'object',
      properties: {
        region: { type: 'string', enum: [...REGIONS, 'all'], description: 'Where.' },
        level: unit('0 released .. 1 fully held.'),
      },
      required: ['region', 'level'],
      additionalProperties: false,
    },
  },
  {
    name: 'breathe',
    title: 'Breathe',
    description: 'Shape the breath. "natural": the body\'s own rhythm; "abdominal": the belly leads; "reverse": the '
      + 'belly draws in on the inhale; "hold": hold after the current phase for `hold` seconds, or until another breathe. '
      + 'Give inhale/exhale/pause (seconds) for a paced breath; nose false breathes through the mouth.',
    inputSchema: {
      type: 'object',
      properties: {
        pattern: { type: 'string', enum: ['natural', 'abdominal', 'reverse', 'hold'], description: 'The kind of breath.' },
        inhale: seconds('Seconds per inhale (paced breath).'),
        exhale: seconds('Seconds per exhale (paced breath).'),
        pause: seconds('Seconds of pause after the exhale.'),
        hold: seconds('With pattern "hold": seconds to hold; leave out to hold until the next breathe.'),
        nose: { type: 'boolean', description: 'Breathe through the nose (true) or the mouth (false).' },
      },
      additionalProperties: false,
    },
  },
  {
    name: 'form',
    title: 'Play a form',
    description: 'Play a named sequence of intentions. Progress arrives as body signals (step by step, then done) '
      + 'when the host grants push events; sense shows where the body is at any moment.',
    inputSchema: {
      type: 'object',
      properties: { name: { type: 'string', enum: ['wuji', 'embrace_tree', 'lift_sky'], description: 'Which form.' } },
      required: ['name'],
      additionalProperties: false,
    },
  },
  {
    name: 'reset',
    title: 'Return to wuji',
    description: 'Return to the wuji stance, natural breath, all tension released.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'viewer',
    title: 'Open the viewer',
    description: 'Start (or find) the viewer: a local web page where someone can watch the practice from outside: the '
      + 'bones moving, the muscles working, the weight in the feet, the breath and the heart. Returns its URL '
      + '(127.0.0.1) to share. The felt sense stays private; the page shows only what the body does.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
];

export const TOOL_NAMES: ReadonlySet<string> = new Set(TOOLS.map((t) => t.name));

/** Only the arguments a tool declares go to the engine; the engine judges their values. */
export function engineArgs(tool: ToolDefinition, raw: unknown): { args: Record<string, unknown>; ignored: string[] } {
  const given = typeof raw === 'object' && raw !== null && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
  const args: Record<string, unknown> = {};
  const ignored: string[] = [];
  for (const [k, v] of Object.entries(given)) {
    if (Object.hasOwn(tool.inputSchema.properties, k)) args[k] = v;
    else ignored.push(k);
  }
  return { args, ignored };
}

export type ToolResult = { content: Array<{ type: 'text'; text: string }>; isError?: boolean };

/** The engine's prose first, then the rest of its result as compact JSON. */
export function resultContent(result: Record<string, unknown>, note?: string): ToolResult {
  const { text, ...rest } = result;
  const content: ToolResult['content'] = [];
  const prose = typeof text === 'string' ? text : '';
  if (prose || note) content.push({ type: 'text', text: [prose, note].filter(Boolean).join('\n') });
  if (Object.keys(rest).length > 0) content.push({ type: 'text', text: JSON.stringify(rest) });
  if (content.length === 0) content.push({ type: 'text', text: '{}' });
  return { content };
}

export function errorResult(text: string): ToolResult {
  return { content: [{ type: 'text', text }], isError: true };
}

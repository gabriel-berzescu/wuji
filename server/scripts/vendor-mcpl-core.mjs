#!/usr/bin/env node
// Refresh the vendored copy of @animalabs/mcpl-core (anima-research/mcpl-core-ts, src/) from a
// local checkout, the way mcpl-cc-bridge vendors it. The copy is built by this package's own tsc,
// so the server needs no registry access to @animalabs and always runs the source of the commit
// recorded in src/vendor/mcpl-core/VENDOR.json.
//
//   node scripts/vendor-mcpl-core.mjs [path/to/mcpl-core-ts]
//
// The checkout defaults to $MCPL_CORE_TS, then ../../mcpl-core-ts (a sibling of the wuji repo).
// Only the package root's modules are copied (src/*.ts); the toolsets entry point is not used here.

import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const serverDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(process.argv[2] ?? process.env.MCPL_CORE_TS ?? join(serverDir, '..', '..', 'mcpl-core-ts'));
const srcDir = join(source, 'src');
const destDir = join(serverDir, 'src', 'vendor', 'mcpl-core');

if (!existsSync(join(srcDir, 'index.ts'))) {
  console.error(`vendor-mcpl-core: no mcpl-core-ts checkout at ${source} (expected src/index.ts).`);
  console.error('Clone https://github.com/anima-research/mcpl-core-ts beside the wuji repo, or pass its path.');
  process.exit(1);
}

const files = readdirSync(srcDir).filter((f) => f.endsWith('.ts') && !f.endsWith('.d.ts')).sort();
rmSync(destDir, { recursive: true, force: true });
mkdirSync(destDir, { recursive: true });
for (const f of files) copyFileSync(join(srcDir, f), join(destDir, f));
// mcpl-core is MIT (Anima Research): its license travels with the copy
if (existsSync(join(source, 'LICENSE'))) copyFileSync(join(source, 'LICENSE'), join(destDir, 'LICENSE'));

const git = (...args) => {
  const r = spawnSync('git', ['-C', source, ...args], { encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : null;
};
const pkg = JSON.parse(readFileSync(join(source, 'package.json'), 'utf8'));
const record = {
  package: pkg.name,
  version: pkg.version,
  repository: 'https://github.com/anima-research/mcpl-core-ts',
  commit: git('rev-parse', 'HEAD'),
  committed: git('log', '-1', '--format=%cI'),
  dirty: (git('status', '--porcelain', '--', 'src') ?? '') !== '',
  files,
};
writeFileSync(join(destDir, 'VENDOR.json'), JSON.stringify(record, null, 2) + '\n');
console.log(`vendored ${record.package}@${record.version} (${record.commit ?? 'unknown commit'}${record.dirty ? ', with local changes' : ''}): ${files.length} files`);

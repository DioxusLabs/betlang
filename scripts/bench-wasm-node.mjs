#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

const modulePath =
  process.argv.find((arg) => !arg.startsWith("--") && arg.endsWith(".wasm")) ??
  "target/wasm32-unknown-unknown/release/examples/wasm_bench_guest.wasm";

const options = {
  samples: optionNumber("--samples", 10),
  sampleMs: optionNumber("--sample-ms", 500),
  warmupMs: optionNumber("--warmup-ms", 1000),
  shortBatch: optionNumber("--short-iters", 64),
  fullBatch: optionNumber("--full-iters", 4),
};

const wasm = await WebAssembly.instantiate(await readFile(modulePath), {});
const exports = wasm.instance.exports;

assert(exports.detect_short_bench(1) === 1, "short benchmark sanity check failed");
assert(
  exports.detect_full_window_bench(1) === 1,
  "full-window benchmark sanity check failed",
);

console.log("runtime: node");
console.log(`module: ${modulePath}`);
console.log(
  `warmup: ${options.warmupMs} ms, sample: ${options.sampleMs} ms, samples: ${options.samples}`,
);
console.log();
console.log(
  `${"case".padEnd(12)} ${"bytes".padStart(9)} ${"batch".padStart(8)} ${"median ns/inf".padStart(14)} ${"inf/s".padStart(12)} ${"MB/s".padStart(12)} ${"best ns/inf".padStart(14)}`,
);

runCase("short", exports.detect_short_bench, exports.detect_short_len(), options.shortBatch);
runCase(
  "full_window",
  exports.detect_full_window_bench,
  exports.detect_full_window_len(),
  options.fullBatch,
);

function runCase(name, detect, bytes, batch) {
  warmup(detect, batch, options.warmupMs);

  const samples = [];
  for (let i = 0; i < options.samples; i++) {
    samples.push(sample(detect, batch, options.sampleMs));
  }

  const ns = samples.map((entry) => entry.nsPerInference).sort((a, b) => a - b);
  const median = ns[Math.floor(ns.length / 2)];
  const best = ns[0];
  const infs = 1_000_000_000 / median;
  const mb = (bytes * infs) / 1_000_000;

  console.log(
    `${name.padEnd(12)} ${String(bytes).padStart(9)} ${String(batch).padStart(8)} ${median.toFixed(0).padStart(14)} ${infs.toFixed(2).padStart(12)} ${mb.toFixed(3).padStart(12)} ${best.toFixed(0).padStart(14)}`,
  );
}

function warmup(detect, batch, ms) {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    assert(detect(batch) === batch, "warmup result mismatch");
  }
}

function sample(detect, batch, ms) {
  const start = performance.now();
  let inferences = 0;
  do {
    assert(detect(batch) === batch, "sample result mismatch");
    inferences += batch;
  } while (performance.now() - start < ms);

  const elapsedNs = (performance.now() - start) * 1_000_000;
  return { nsPerInference: elapsedNs / inferences };
}

function optionNumber(name, fallback) {
  const prefix = `${name}=`;
  const inline = process.argv.find((arg) => arg.startsWith(prefix));
  if (inline) {
    return Number(inline.slice(prefix.length));
  }

  const index = process.argv.indexOf(name);
  if (index !== -1 && process.argv[index + 1]) {
    return Number(process.argv[index + 1]);
  }

  return fallback;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

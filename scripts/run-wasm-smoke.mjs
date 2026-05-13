import { readFile } from "node:fs/promises";

const exportsToRun = [
  "detect_empty_smoke",
  "detect_rust_smoke",
  "detect_full_window_smoke",
];

const paths = process.argv.slice(2);
if (paths.length === 0) {
  throw new Error("usage: node scripts/run-wasm-smoke.mjs <module.wasm>...");
}

for (const path of paths) {
  const bytes = await readFile(path);
  const module = await WebAssembly.compile(bytes);
  const instance = await WebAssembly.instantiate(module, {});

  for (const name of exportsToRun) {
    const exported = instance.exports[name];
    if (typeof exported !== "function") {
      throw new Error(`${path}: missing export ${name}`);
    }

    const status = exported();
    if (status !== 0) {
      throw new Error(`${path}: ${name} returned ${status}`);
    }
  }

  console.log(`${path}: wasm smoke passed`);
}

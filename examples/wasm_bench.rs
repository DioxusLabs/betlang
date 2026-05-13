use std::path::PathBuf;
use std::time::{Duration, Instant};

use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

const DEFAULT_WASM_MODULE: &str = "target/wasm32-unknown-unknown/release/examples/wasm_smoke.wasm";
const DEFAULT_SAMPLES: usize = 10;
const DEFAULT_SAMPLE_MS: u64 = 500;
const DEFAULT_WARMUP_MS: u64 = 1_000;
const DEFAULT_SHORT_BATCH: u32 = 64;
const DEFAULT_FULL_WINDOW_BATCH: u32 = 4;

#[derive(Clone)]
struct Options {
    module: PathBuf,
    samples: usize,
    sample_duration: Duration,
    warmup_duration: Duration,
    short_batch: u32,
    full_window_batch: u32,
}

struct WasmDetect {
    store: Store<()>,
    short: TypedFunc<u32, u32>,
    full_window: TypedFunc<u32, u32>,
    short_len: u32,
    full_window_len: u32,
}

#[derive(Clone, Copy)]
enum Case {
    Short,
    FullWindow,
}

struct Sample {
    elapsed: Duration,
    inferences: u64,
    calls: u64,
}

fn main() {
    let options = Options::parse();
    let mut wasm = WasmDetect::load(&options.module);

    println!("runtime: wasmtime");
    println!("module: {}", options.module.display());
    println!(
        "warmup: {} ms, sample: {} ms, samples: {}",
        options.warmup_duration.as_millis(),
        options.sample_duration.as_millis(),
        options.samples
    );
    println!();
    println!(
        "{:<12} {:>9} {:>8} {:>14} {:>12} {:>12} {:>14}",
        "case", "bytes", "batch", "median ns/inf", "inf/s", "MB/s", "best ns/inf"
    );

    run_case(&mut wasm, Case::Short, options.short_batch, &options);
    run_case(
        &mut wasm,
        Case::FullWindow,
        options.full_window_batch,
        &options,
    );
}

impl Options {
    fn parse() -> Self {
        let mut module = std::env::var_os("BETLANG_WASM_MODULE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WASM_MODULE));
        let mut module_was_set = false;
        let mut samples = DEFAULT_SAMPLES;
        let mut sample_ms = DEFAULT_SAMPLE_MS;
        let mut warmup_ms = DEFAULT_WARMUP_MS;
        let mut short_batch = DEFAULT_SHORT_BATCH;
        let mut full_window_batch = DEFAULT_FULL_WINDOW_BATCH;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--samples" => samples = parse_arg(args.next(), "--samples"),
                "--sample-ms" => sample_ms = parse_arg(args.next(), "--sample-ms"),
                "--warmup-ms" => warmup_ms = parse_arg(args.next(), "--warmup-ms"),
                "--short-iters" => short_batch = parse_arg(args.next(), "--short-iters"),
                "--full-iters" => {
                    full_window_batch = parse_arg(args.next(), "--full-iters");
                }
                _ if arg.starts_with("--samples=") => {
                    samples = parse_value(&arg["--samples=".len()..], "--samples");
                }
                _ if arg.starts_with("--sample-ms=") => {
                    sample_ms = parse_value(&arg["--sample-ms=".len()..], "--sample-ms");
                }
                _ if arg.starts_with("--warmup-ms=") => {
                    warmup_ms = parse_value(&arg["--warmup-ms=".len()..], "--warmup-ms");
                }
                _ if arg.starts_with("--short-iters=") => {
                    short_batch = parse_value(&arg["--short-iters=".len()..], "--short-iters");
                }
                _ if arg.starts_with("--full-iters=") => {
                    full_window_batch = parse_value(&arg["--full-iters=".len()..], "--full-iters");
                }
                _ if arg.starts_with('-') => panic!("unknown option {arg}"),
                _ => {
                    assert!(!module_was_set, "multiple wasm module paths supplied");
                    module = PathBuf::from(arg);
                    module_was_set = true;
                }
            }
        }

        assert!(samples > 0, "--samples must be greater than zero");
        assert!(sample_ms > 0, "--sample-ms must be greater than zero");
        assert!(short_batch > 0, "--short-iters must be greater than zero");
        assert!(
            full_window_batch > 0,
            "--full-iters must be greater than zero"
        );

        Self {
            module,
            samples,
            sample_duration: Duration::from_millis(sample_ms),
            warmup_duration: Duration::from_millis(warmup_ms),
            short_batch,
            full_window_batch,
        }
    }
}

impl WasmDetect {
    fn load(path: &PathBuf) -> Self {
        let engine = Engine::default();
        let module = Module::from_file(&engine, path)
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", path.display()));
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).expect("instantiate wasm module");

        let short = typed_func(&mut store, &instance, "detect_short_bench");
        let full_window = typed_func(&mut store, &instance, "detect_full_window_bench");
        let short_len = typed_func::<(), u32>(&mut store, &instance, "detect_short_len")
            .call(&mut store, ())
            .expect("call detect_short_len");
        let full_window_len =
            typed_func::<(), u32>(&mut store, &instance, "detect_full_window_len")
                .call(&mut store, ())
                .expect("call detect_full_window_len");

        let mut this = Self {
            store,
            short,
            full_window,
            short_len,
            full_window_len,
        };
        this.assert_correct();
        this
    }

    fn assert_correct(&mut self) {
        assert_eq!(self.call(Case::Short, 1), 1);
        assert_eq!(self.call(Case::FullWindow, 1), 1);
    }

    fn call(&mut self, case: Case, iterations: u32) -> u32 {
        let function = match case {
            Case::Short => &self.short,
            Case::FullWindow => &self.full_window,
        };
        function
            .call(&mut self.store, iterations)
            .expect("call wasm benchmark export")
    }

    fn bytes_per_inference(&self, case: Case) -> u32 {
        match case {
            Case::Short => self.short_len,
            Case::FullWindow => self.full_window_len,
        }
    }
}

fn typed_func<Params, Results>(
    store: &mut Store<()>,
    instance: &Instance,
    name: &str,
) -> TypedFunc<Params, Results>
where
    Params: wasmtime::WasmParams,
    Results: wasmtime::WasmResults,
{
    instance
        .get_typed_func(store, name)
        .unwrap_or_else(|error| panic!("missing wasm export {name}: {error}"))
}

fn run_case(wasm: &mut WasmDetect, case: Case, batch: u32, options: &Options) {
    warm_up(wasm, case, batch, options.warmup_duration);

    let mut samples = Vec::with_capacity(options.samples);
    for _ in 0..options.samples {
        samples.push(sample_case(wasm, case, batch, options.sample_duration));
    }

    let bytes = wasm.bytes_per_inference(case) as f64;
    let median_ns = median_ns_per_inference(&samples);
    let best_ns = samples
        .iter()
        .map(ns_per_inference)
        .fold(f64::INFINITY, f64::min);
    let median_inf_s = 1_000_000_000.0 / median_ns;
    let median_mb_s = bytes * median_inf_s / 1_000_000.0;
    let best_calls = samples.iter().map(|sample| sample.calls).max().unwrap_or(0);

    println!(
        "{:<12} {:>9} {:>8} {:>14.0} {:>12.2} {:>12.3} {:>14.0}  # max calls/sample: {}",
        case.name(),
        bytes as u32,
        batch,
        median_ns,
        median_inf_s,
        median_mb_s,
        best_ns,
        best_calls
    );
}

fn warm_up(wasm: &mut WasmDetect, case: Case, batch: u32, duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        assert_eq!(wasm.call(case, batch), batch);
    }
}

fn sample_case(wasm: &mut WasmDetect, case: Case, batch: u32, duration: Duration) -> Sample {
    let start = Instant::now();
    let mut inferences = 0;
    let mut calls = 0;
    loop {
        assert_eq!(wasm.call(case, batch), batch);
        inferences += u64::from(batch);
        calls += 1;
        if start.elapsed() >= duration {
            break;
        }
    }

    Sample {
        elapsed: start.elapsed(),
        inferences,
        calls,
    }
}

fn median_ns_per_inference(samples: &[Sample]) -> f64 {
    let mut values = samples.iter().map(ns_per_inference).collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn ns_per_inference(sample: &Sample) -> f64 {
    sample.elapsed.as_secs_f64() * 1_000_000_000.0 / sample.inferences as f64
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::Short => "short",
            Case::FullWindow => "full_window",
        }
    }
}

fn parse_arg<T>(value: Option<String>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value.unwrap_or_else(|| panic!("{name} requires a value"));
    parse_value(&value, name)
}

fn parse_value<T>(value: &str, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid {name} value {value:?}: {error}"))
}

fn print_usage() {
    println!(
        "usage: cargo run --release --features wasm-bench --example wasm_bench -- [module.wasm] [options]\n\
         options:\n\
         \t--samples N       default {DEFAULT_SAMPLES}\n\
         \t--sample-ms N     default {DEFAULT_SAMPLE_MS}\n\
         \t--warmup-ms N     default {DEFAULT_WARMUP_MS}\n\
         \t--short-iters N   guest iterations per host call, default {DEFAULT_SHORT_BATCH}\n\
         \t--full-iters N    guest iterations per host call, default {DEFAULT_FULL_WINDOW_BATCH}"
    );
}

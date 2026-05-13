use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const MAGIKA_BLOCK_SIZE: usize = 4_096;

fn bench_detect(c: &mut Criterion) {
    let short = include_str!("../snippets/demo.rs");
    let full = full_window_source(short);

    let mut group = c.benchmark_group("detect");
    for (name, source) in [("short", short), ("full_window", full.as_str())] {
        assert_eq!(betlang::detect(source), Some(betlang::Language::Rust));
        group.throughput(Throughput::BytesDecimal(source.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), source, |b, source| {
            b.iter(|| black_box(betlang::detect(black_box(source))));
        });
    }
    group.finish();
}

fn full_window_source(seed: &str) -> String {
    let mut source = String::new();
    while source.len() < MAGIKA_BLOCK_SIZE + 512 {
        source.push_str(seed);
        source.push('\n');
    }
    source
}

criterion_group!(benches, bench_detect);
criterion_main!(benches);

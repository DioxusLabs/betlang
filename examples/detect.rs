//! Detect the language of a file, of stdin, or — given a directory — print
//! every file under the tree with its detected language and a GitHub-style
//! breakdown.
//!
//! ```text
//! cargo run --release --example detect -- src/model.rs   # single file
//! cargo run --release --example detect < snippets/demo.rs # stdin
//! cargo run --release --example detect -- .              # tree breakdown
//! ```
//!
//! Tree mode walks the path with the [`ignore`] crate, so `.gitignore` and
//! `.git/` are respected by default (matching what `git ls-files` would show).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use betlang::Language;
use rayon::prelude::*;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let arg = args.next();

    if args.next().is_some() {
        eprintln!("usage: detect [PATH]   (omit PATH to read stdin)");
        return ExitCode::from(2);
    }

    match arg.as_deref() {
        None => detect_stdin(),
        Some(path) => {
            let meta = match fs::metadata(path) {
                Ok(meta) => meta,
                Err(err) => {
                    eprintln!("betlang: failed to stat {path}: {err}");
                    return ExitCode::from(2);
                }
            };
            if meta.is_dir() {
                breakdown_tree(Path::new(path))
            } else {
                detect_file(Path::new(path))
            }
        }
    }
}

fn detect_stdin() -> ExitCode {
    let mut buf = Vec::new();
    if let Err(err) = io::stdin().read_to_end(&mut buf) {
        eprintln!("betlang: failed to read stdin: {err}");
        return ExitCode::from(2);
    }
    report_single(betlang::detect(&buf))
}

fn detect_file(path: &Path) -> ExitCode {
    let (bytes, _) = match read_model_window(path) {
        Ok(window) => window,
        Err(err) => {
            eprintln!("betlang: failed to read {}: {err}", path.display());
            return ExitCode::from(2);
        }
    };
    report_single(betlang::detect(bytes))
}

/// The model only inspects the first and last 4096 bytes of a file, so read
/// just those two windows instead of the whole file. `build_window` derives
/// its begin/end blocks from `source[..4096]` and `source[len - 4096..]`,
/// which the concatenated head + tail preserves exactly. Returns the window
/// bytes and the file's real size.
fn read_model_window(path: &Path) -> io::Result<(Vec<u8>, u64)> {
    const BLOCK: u64 = 4096;

    let mut file = fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size <= 2 * BLOCK {
        let mut bytes = Vec::with_capacity(size as usize);
        file.read_to_end(&mut bytes)?;
        return Ok((bytes, size));
    }

    let mut bytes = vec![0u8; (2 * BLOCK) as usize];
    file.read_exact(&mut bytes[..BLOCK as usize])?;
    file.seek(SeekFrom::End(-(BLOCK as i64)))?;
    file.read_exact(&mut bytes[BLOCK as usize..])?;
    Ok((bytes, size))
}

fn report_single(detection: betlang::Detection) -> ExitCode {
    match detection.language() {
        Some(language) => {
            println!("{} ({:?})", language.slug(), language);
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("betlang: no match");
            ExitCode::from(1)
        }
    }
}

enum Kind {
    Dir,
    File {
        language: Option<Language>,
        truth: Option<Language>,
        size: u64,
    },
    Unreadable,
}

enum PendingKind {
    Dir,
    File {
        path: PathBuf,
        truth: Option<Language>,
    },
}

struct PendingNode {
    depth: usize,
    name: String,
    kind: PendingKind,
}

struct Node {
    depth: usize,
    name: String,
    kind: Kind,
}

#[derive(Default, Clone, Copy)]
struct LangStat {
    correct: u64,
    total: u64,
}

fn breakdown_tree(root: &Path) -> ExitCode {
    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .require_git(false)
        .sort_by_file_path(|a, b| a.cmp(b))
        .build();

    let mut pending_nodes: Vec<PendingNode> = Vec::new();
    let mut bytes_by_language: HashMap<Language, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut undetected: u64 = 0;
    let mut confusion: HashMap<(Language, Option<Language>), u64> = HashMap::new();
    let mut by_truth: HashMap<Language, LangStat> = HashMap::new();
    let mut graded: u64 = 0;
    let mut correct: u64 = 0;

    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!("betlang: walk error: {err}");
                continue;
            }
        };
        let depth = entry.depth();
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
        let name = display_name(entry.path(), root, depth);

        if is_dir {
            pending_nodes.push(PendingNode {
                depth,
                name,
                kind: PendingKind::Dir,
            });
            continue;
        }

        pending_nodes.push(PendingNode {
            depth,
            name,
            kind: PendingKind::File {
                path: entry.path().to_path_buf(),
                truth: ground_truth(entry.path()),
            },
        });
    }

    let nodes: Vec<Node> = pending_nodes.into_par_iter().map(classify_node).collect();

    if nodes.is_empty() {
        eprintln!("betlang: nothing to scan under {}", root.display());
        return ExitCode::from(1);
    }

    for node in &nodes {
        let Kind::File {
            language,
            truth,
            size,
        } = node.kind
        else {
            continue;
        };

        total += size;
        match language {
            Some(language) => *bytes_by_language.entry(language).or_default() += size,
            None => undetected += size,
        }
        if let Some(truth_lang) = truth {
            graded += 1;
            let stat = by_truth.entry(truth_lang).or_default();
            stat.total += 1;
            if language == Some(truth_lang) {
                stat.correct += 1;
                correct += 1;
            }
            *confusion.entry((truth_lang, language)).or_default() += 1;
        }
    }

    print_tree(&nodes);

    if total > 0 {
        println!();
        println!("Breakdown:");
        print_breakdown(&bytes_by_language, total, undetected);
    }

    if graded > 0 {
        println!();
        print_accuracy(&by_truth, graded, correct);
        println!();
        print_confusion(&by_truth, &confusion);
    }

    ExitCode::SUCCESS
}

fn classify_node(node: PendingNode) -> Node {
    let kind = match node.kind {
        PendingKind::Dir => Kind::Dir,
        PendingKind::File { path, truth } => match classify_file(&path) {
            Some((language, size)) => Kind::File {
                language,
                truth,
                size,
            },
            None => Kind::Unreadable,
        },
    };

    Node {
        depth: node.depth,
        name: node.name,
        kind,
    }
}

fn display_name(path: &Path, root: &Path, depth: usize) -> String {
    if depth == 0 {
        return root.display().to_string();
    }
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn classify_file(path: &Path) -> Option<(Option<Language>, u64)> {
    let (bytes, size) = read_model_window(path).ok()?;
    let language = betlang::detect(bytes).language();
    Some((language, size))
}

fn print_tree(nodes: &[Node]) {
    let is_last = compute_is_last(nodes);
    let prefixes = compute_prefixes(nodes, &is_last);

    let label_width = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !matches!(node.kind, Kind::Dir) && node.depth > 0)
        .map(|(i, node)| prefixes[i].chars().count() + node.name.chars().count())
        .max()
        .unwrap_or(0);

    for (i, node) in nodes.iter().enumerate() {
        if node.depth == 0 {
            println!("{}", node.name);
            continue;
        }
        let prefix = &prefixes[i];
        match &node.kind {
            Kind::Dir => println!("{prefix}{}/", node.name),
            Kind::File {
                language,
                truth,
                size,
            } => {
                let label = format!("{prefix}{}", node.name);
                let pad = label_width.saturating_sub(label.chars().count()) + 2;
                let tag = language
                    .map(|language| language.slug().to_string())
                    .unwrap_or_else(|| "?".into());
                let mark = match truth {
                    Some(t) if *language == Some(*t) => " ✓".to_string(),
                    Some(t) => format!(" ✗ (expected {})", t.slug()),
                    None => String::new(),
                };
                println!(
                    "{label}{spaces}{tag}  ({size}){mark}",
                    spaces = " ".repeat(pad),
                    size = format_bytes(*size),
                );
            }
            Kind::Unreadable => {
                let label = format!("{prefix}{}", node.name);
                let pad = label_width.saturating_sub(label.chars().count()) + 2;
                println!("{label}{}(unreadable)", " ".repeat(pad));
            }
        }
    }
}

fn compute_is_last(nodes: &[Node]) -> Vec<bool> {
    let mut is_last = vec![false; nodes.len()];
    for i in 0..nodes.len() {
        let d = nodes[i].depth;
        let mut last = true;
        for next in &nodes[i + 1..] {
            if next.depth < d {
                break;
            }
            if next.depth == d {
                last = false;
                break;
            }
        }
        is_last[i] = last;
    }
    is_last
}

/// For each node, build its `│   `/`    ` ancestor columns followed by a
/// `├── ` or `└── ` connector, in a single forward pass over the tree.
fn compute_prefixes(nodes: &[Node], is_last: &[bool]) -> Vec<String> {
    let mut prefixes = Vec::with_capacity(nodes.len());
    // `stack[k]` describes the column drawn for an entry whose ancestor at
    // depth `k + 1` is "non-last" (`│   `) or "last" (`    `).
    let mut stack: Vec<&'static str> = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.depth == 0 {
            prefixes.push(String::new());
            stack.clear();
            continue;
        }
        // Trim the stack to the ancestor columns that apply at this depth.
        stack.truncate(node.depth - 1);

        let mut prefix = String::with_capacity(node.depth * 4);
        for column in &stack {
            prefix.push_str(column);
        }
        prefix.push_str(if is_last[i] {
            "└── "
        } else {
            "├── "
        });
        prefixes.push(prefix);

        // Push the column descendants of this entry will draw underneath us.
        stack.push(if is_last[i] { "    " } else { "│   " });
    }
    prefixes
}

fn print_breakdown(bytes_by_language: &HashMap<Language, u64>, total: u64, undetected: u64) {
    let mut ranked: Vec<(Language, u64)> = bytes_by_language
        .iter()
        .map(|(language, size)| (*language, *size))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.slug().cmp(b.0.slug())));

    let name_width = ranked
        .iter()
        .map(|(language, _)| language.slug().len())
        .max()
        .unwrap_or(0)
        .max("(undetected)".len());

    for (language, size) in &ranked {
        let pct = 100.0 * (*size as f64) / (total as f64);
        println!(
            "  {:<width$}  {:>6.2}%  {}",
            language.slug(),
            pct,
            format_bytes(*size),
            width = name_width,
        );
    }
    if undetected > 0 {
        let pct = 100.0 * (undetected as f64) / (total as f64);
        println!(
            "  {:<width$}  {:>6.2}%  {}",
            "(undetected)",
            pct,
            format_bytes(undetected),
            width = name_width,
        );
    }
}

fn print_accuracy(by_truth: &HashMap<Language, LangStat>, graded: u64, correct: u64) {
    let pct = 100.0 * (correct as f64) / (graded as f64);
    println!("Accuracy: {pct:.2}%  ({correct}/{graded} files with known extension)");

    let mut rows: Vec<(Language, LangStat)> = by_truth.iter().map(|(l, s)| (*l, *s)).collect();
    rows.sort_by(|a, b| {
        b.1.total
            .cmp(&a.1.total)
            .then_with(|| a.0.slug().cmp(b.0.slug()))
    });

    let name_width = rows
        .iter()
        .map(|(language, _)| language.slug().len())
        .max()
        .unwrap_or(0);

    println!();
    println!("By language (truth → correct / total):");
    for (language, stat) in &rows {
        let pct = 100.0 * (stat.correct as f64) / (stat.total as f64);
        println!(
            "  {:<width$}  {:>6.2}%  ({}/{})",
            language.slug(),
            pct,
            stat.correct,
            stat.total,
            width = name_width,
        );
    }
}

fn print_confusion(
    by_truth: &HashMap<Language, LangStat>,
    confusion: &HashMap<(Language, Option<Language>), u64>,
) {
    let mut truths: Vec<Language> = by_truth.keys().copied().collect();
    truths.sort_by_key(|l| l.slug());

    // Predicted columns: every language that shows up as a prediction for any
    // truth row, plus a final `(none)` column for undetected.
    let mut predictions: Vec<Language> = confusion
        .keys()
        .filter_map(|(_, pred)| *pred)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    predictions.sort_by_key(|l| l.slug());
    let has_undetected = confusion.keys().any(|(_, pred)| pred.is_none());

    let truth_width = truths
        .iter()
        .map(|l| l.slug().len())
        .max()
        .unwrap_or(0)
        .max("truth\\pred".len());

    // Column width chosen to fit the longest header or three-digit count.
    let col_width = predictions
        .iter()
        .map(|l| l.slug().len())
        .max()
        .unwrap_or(0)
        .max(if has_undetected { "(none)".len() } else { 0 })
        .max(4);

    println!("Confusion matrix (rows = truth from extension, cols = predicted):");
    print!("  {:<width$}", "truth\\pred", width = truth_width);
    for pred in &predictions {
        print!("  {:>width$}", pred.slug(), width = col_width);
    }
    if has_undetected {
        print!("  {:>width$}", "(none)", width = col_width);
    }
    println!();

    for truth in &truths {
        print!("  {:<width$}", truth.slug(), width = truth_width);
        for pred in &predictions {
            let count = confusion.get(&(*truth, Some(*pred))).copied().unwrap_or(0);
            if count == 0 {
                print!("  {:>width$}", ".", width = col_width);
            } else {
                print!("  {count:>width$}", width = col_width);
            }
        }
        if has_undetected {
            let count = confusion.get(&(*truth, None)).copied().unwrap_or(0);
            if count == 0 {
                print!("  {:>width$}", ".", width = col_width);
            } else {
                print!("  {count:>width$}", width = col_width);
            }
        }
        println!();
    }
}

/// Filename- and extension-based ground truth. Returns `None` for files whose
/// extension is ambiguous, outside the model output surface, or unknown, so
/// those files are excluded from accuracy and confusion counts.
fn ground_truth(path: &Path) -> Option<Language> {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        match name {
            "Dockerfile" | "Containerfile" | "dockerfile" => return Some(Language::Dockerfile),
            "CMakeLists.txt" => return Some(Language::CMake),
            "Gemfile" => return Some(Language::Gemfile),
            "Makefile" | "GNUmakefile" | "makefile" => return None, // not a tracked language
            "BUILD" | "BUILD.bazel" | "WORKSPACE" | "WORKSPACE.bazel" => return None,
            _ => {}
        }
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "asm" | "s" => Language::Asm,
        "bat" | "cmd" => Language::Batch,
        "sh" | "bash" | "zsh" | "ksh" => Language::Shell,
        "c" => Language::C,
        "cs" => Language::Cs,
        "clj" | "cljs" | "cljc" | "edn" => Language::Clojure,
        "cmake" => Language::CMake,
        "cob" | "cbl" | "cpy" => Language::Cobol,
        "lisp" | "lsp" | "cl" => Language::Lisp,
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Language::Cpp,
        "css" => Language::Css,
        "dart" => Language::Dart,
        "ex" | "exs" => Language::Elixir,
        "erl" | "hrl" => Language::Erlang,
        "gemspec" => Language::Gemspec,
        "go" => Language::Go,
        "gradle" => Language::Gradle,
        "groovy" | "gvy" => Language::Groovy,
        "hs" | "lhs" => Language::Haskell,
        "html" | "htm" | "xhtml" => Language::Html,
        "ini" | "cfg" => Language::Ini,
        "java" => Language::Java,
        "js" | "mjs" | "cjs" | "jsx" => Language::JavaScript,
        "json" | "jsonc" | "json5" => Language::Json,
        "jl" => Language::Julia,
        "kt" | "kts" => Language::Kotlin,
        "lua" => Language::Lua,
        "md" | "markdown" | "mdx" => Language::Markdown,
        "mm" => Language::ObjectiveC,
        "ml" | "mli" => Language::Ocaml,
        "php" | "phtml" => Language::Php,
        "ps1" | "psm1" | "psd1" => Language::Powershell,
        "py" | "pyi" | "pyw" => Language::Python,
        "rb" | "rake" => Language::Ruby,
        "rs" => Language::Rust,
        "scala" | "sbt" => Language::Scala,
        "sql" => Language::Sql,
        "swift" => Language::Swift,
        "toml" => Language::Toml,
        "ts" | "tsx" | "mts" | "cts" => Language::TypeScript,
        "vb" | "vbs" => Language::Vba,
        "v" | "sv" | "svh" => Language::Verilog,
        "xml" | "xsd" | "xsl" | "xslt" | "svg" | "plist" => Language::Xml,
        "yaml" | "yml" => Language::Yaml,
        // Ambiguous, unsupported, or out-of-vocabulary: skip.
        // Examples: .m, .pl, .h, .r.
        _ => return None,
    })
}

fn format_bytes(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = size as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

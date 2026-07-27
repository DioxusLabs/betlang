//! Classify a file, stdin, or — given a directory — every file under the
//! tree as natural language or an LLM prompt, with a breakdown.
//!
//! ```text
//! cargo run --release --example detect -- notes.txt              # single file
//! cargo run --release --example detect < snippets/demo-prompt.txt # stdin
//! cargo run --release --example detect -- .                      # tree breakdown
//! ```
//!
//! Tree mode walks the path with the [`ignore`] crate, so `.gitignore` and
//! `.git/` are respected by default (matching what `git ls-files` would show).

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use betlang::Kind;
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
    match detection.top_kinds().next() {
        Some((probability, kind)) => {
            println!("{} ({probability:.3})", kind.slug());
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("betlang: no match");
            ExitCode::from(1)
        }
    }
}

enum NodeKind {
    Dir,
    File { kind: Option<Kind>, size: u64 },
    Unreadable,
}

enum PendingNodeKind {
    Dir,
    File { path: PathBuf },
}

struct PendingNode {
    depth: usize,
    name: String,
    kind: PendingNodeKind,
}

struct Node {
    depth: usize,
    name: String,
    kind: NodeKind,
}

fn breakdown_tree(root: &Path) -> ExitCode {
    let walker = ignore::WalkBuilder::new(root)
        .standard_filters(true)
        .hidden(true)
        .require_git(false)
        .sort_by_file_path(|a, b| a.cmp(b))
        .build();

    let mut pending_nodes: Vec<PendingNode> = Vec::new();
    let mut bytes_by_kind: HashMap<Kind, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut undetected: u64 = 0;

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
                kind: PendingNodeKind::Dir,
            });
            continue;
        }

        pending_nodes.push(PendingNode {
            depth,
            name,
            kind: PendingNodeKind::File {
                path: entry.path().to_path_buf(),
            },
        });
    }

    let nodes: Vec<Node> = pending_nodes.into_par_iter().map(classify_node).collect();

    if nodes.is_empty() {
        eprintln!("betlang: nothing to scan under {}", root.display());
        return ExitCode::from(1);
    }

    for node in &nodes {
        let NodeKind::File { kind, size } = node.kind else {
            continue;
        };

        total += size;
        match kind {
            Some(kind) => *bytes_by_kind.entry(kind).or_default() += size,
            None => undetected += size,
        }
    }

    print_tree(&nodes);

    if total > 0 {
        println!();
        println!("Breakdown:");
        print_breakdown(&bytes_by_kind, total, undetected);
    }

    ExitCode::SUCCESS
}

fn classify_node(node: PendingNode) -> Node {
    let kind = match node.kind {
        PendingNodeKind::Dir => NodeKind::Dir,
        PendingNodeKind::File { path } => match classify_file(&path) {
            Some((kind, size)) => NodeKind::File { kind, size },
            None => NodeKind::Unreadable,
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

fn classify_file(path: &Path) -> Option<(Option<Kind>, u64)> {
    let (bytes, size) = read_model_window(path).ok()?;
    let kind = betlang::detect(bytes).kind();
    Some((kind, size))
}

fn print_tree(nodes: &[Node]) {
    let is_last = compute_is_last(nodes);
    let prefixes = compute_prefixes(nodes, &is_last);

    let label_width = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !matches!(node.kind, NodeKind::Dir) && node.depth > 0)
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
            NodeKind::Dir => println!("{prefix}{}/", node.name),
            NodeKind::File { kind, size } => {
                let label = format!("{prefix}{}", node.name);
                let pad = label_width.saturating_sub(label.chars().count()) + 2;
                let tag = kind
                    .map(|kind| kind.slug().to_string())
                    .unwrap_or_else(|| "?".into());
                println!(
                    "{label}{spaces}{tag}  ({size})",
                    spaces = " ".repeat(pad),
                    size = format_bytes(*size),
                );
            }
            NodeKind::Unreadable => {
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

fn print_breakdown(bytes_by_kind: &HashMap<Kind, u64>, total: u64, undetected: u64) {
    let mut ranked: Vec<(Kind, u64)> = bytes_by_kind
        .iter()
        .map(|(kind, size)| (*kind, *size))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.slug().cmp(b.0.slug())));

    let name_width = ranked
        .iter()
        .map(|(kind, _)| kind.slug().len())
        .max()
        .unwrap_or(0)
        .max("(undetected)".len());

    for (kind, size) in &ranked {
        let pct = 100.0 * (*size as f64) / (total as f64);
        println!(
            "  {:<width$}  {:>6.2}%  {}",
            kind.slug(),
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

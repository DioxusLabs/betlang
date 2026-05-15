use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn detects_stdin() {
    let output = run_with_stdin(
        detect_command(),
        "fn main() {\n    println!(\"hello\");\n}\n",
    );

    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("rust"));
}

#[test]
fn reports_no_match_for_whitespace_stdin() {
    let output = run_with_stdin(detect_command(), "  \n\t  ");

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("no match"));
}

#[test]
fn detects_file_path() {
    let temp = TempDir::new("betlang-detect-file");
    let file = temp.path().join("main.rs");
    fs::write(&file, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

    let output = detect_command().arg(file).output().unwrap();

    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("rust"));
}

#[test]
fn rejects_non_utf8_file_path() {
    let temp = TempDir::new("betlang-detect-invalid");
    let file = temp.path().join("invalid.rs");
    fs::write(&file, b"fn main() {\n\xff\xfe\n}\n").unwrap();

    let output = detect_command().arg(&file).output().unwrap();

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("not valid UTF-8"));
}

#[test]
fn rejects_too_many_arguments() {
    let output = detect_command().args(["one", "two"]).output().unwrap();

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: detect"));
}

#[test]
fn prints_tree_breakdown_and_accuracy() {
    let temp = TempDir::new("betlang-detect-tree");
    fs::create_dir(temp.path().join("src")).unwrap();
    fs::write(
        temp.path().join("src").join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("tool.py"),
        "import pathlib\n\ndef main():\n    print(pathlib.Path.cwd())\n",
    )
    .unwrap();

    let output = detect_command().arg(temp.path()).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("Breakdown:"), "{stdout}");
    assert!(stdout.contains("Accuracy:"), "{stdout}");
    assert!(stdout.contains("rust"), "{stdout}");
    assert!(stdout.contains("python"), "{stdout}");
}

fn run_with_stdin(mut command: Command, input: &str) -> std::process::Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn detect_command() -> Command {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_detect") {
        return Command::new(path);
    }

    Command::new(detect_example_binary())
}

fn detect_example_binary() -> PathBuf {
    static DETECT_EXAMPLE: OnceLock<PathBuf> = OnceLock::new();

    DETECT_EXAMPLE
        .get_or_init(|| {
            let path = detect_example_path();
            if !path.exists() {
                let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
                let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
                    .expect("CARGO_MANIFEST_DIR must be set for integration tests");
                let status = Command::new(cargo)
                    .args(["build", "--quiet", "--example", "detect"])
                    .current_dir(manifest_dir)
                    .status()
                    .expect("failed to run cargo build --example detect");
                assert!(status.success(), "cargo build --example detect failed");
                assert!(
                    path.exists(),
                    "cargo build --example detect did not create {}",
                    path.display()
                );
            }
            path
        })
        .clone()
}

fn detect_example_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.file_name().is_some_and(|name| name == "deps") {
        path.pop();
    }
    path.push("examples");
    path.push(format!("detect{}", std::env::consts::EXE_SUFFIX));
    path
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Self {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("{prefix}-{}-{nanos}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

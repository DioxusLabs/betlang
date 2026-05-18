use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn detects_stdin() -> TestResult {
    let output = run_with_stdin(
        detect_command(),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )?;

    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("rust"));
    Ok(())
}

#[test]
fn reports_no_match_for_whitespace_stdin() -> TestResult {
    let output = run_with_stdin(detect_command(), "  \n\t  ")?;

    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("no match"));
    Ok(())
}

#[test]
fn detects_file_path() -> TestResult {
    let temp = TempDir::new("betlang-detect-file")?;
    let file = temp.path().join("main.rs");
    fs::write(&file, "fn main() {\n    println!(\"hello\");\n}\n")?;

    let output = detect_command().arg(file).output()?;

    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("rust"));
    Ok(())
}

#[test]
fn detects_non_utf8_file_path() -> TestResult {
    let temp = TempDir::new("betlang-detect-invalid")?;
    let file = temp.path().join("invalid.rs");
    fs::write(&file, b"fn main() {\n\xff\xfe\n}\n")?;

    let output = detect_command().arg(&file).output()?;

    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("rust"));
    Ok(())
}

#[test]
fn rejects_too_many_arguments() -> TestResult {
    let output = detect_command().args(["one", "two"]).output()?;

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: detect"));
    Ok(())
}

#[test]
fn prints_tree_breakdown_and_accuracy() -> TestResult {
    let temp = TempDir::new("betlang-detect-tree")?;
    fs::create_dir(temp.path().join("src"))?;
    fs::write(
        temp.path().join("src").join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )?;
    fs::write(
        temp.path().join("tool.py"),
        "import pathlib\n\ndef main():\n    print(pathlib.Path.cwd())\n",
    )?;

    let output = detect_command().arg(temp.path()).output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("Breakdown:"), "{stdout}");
    assert!(stdout.contains("Accuracy:"), "{stdout}");
    assert!(stdout.contains("rust"), "{stdout}");
    assert!(stdout.contains("python"), "{stdout}");
    Ok(())
}

fn run_with_stdin(mut command: Command, input: &str) -> io::Result<std::process::Output> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "child stdin missing"))?;
    stdin.write_all(input.as_bytes())?;
    child.wait_with_output()
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
    let mut path = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => panic!("failed to locate current test executable: {err}"),
    };
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
    fn new(prefix: &str) -> io::Result<Self> {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        path.push(format!("{prefix}-{}-{nanos}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self { path })
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

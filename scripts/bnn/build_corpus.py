#!/usr/bin/env python3
"""Corpus builder: reuses betlang's build_finetune_corpus but replaces the
gated the-stack labels (yaml/json/toml/ini/xml/swift/cobol) with GitHub
tarball harvesting, so no HF token is needed."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "repos/betlang/scripts"))
import build_finetune_corpus as bfc

EXTRA_GITHUB = {
    "yaml": (
        [
            "kubernetes/examples", "argoproj/argo-cd", "istio/istio",
            "ansible/ansible", "prometheus/prometheus", "grafana/grafana",
            "helm/examples", "fluxcd/flux2", "kubernetes-sigs/kustomize",
            "open-telemetry/opentelemetry-collector", "goharbor/harbor",
            "jaegertracing/jaeger", "cilium/cilium", "envoyproxy/envoy",
        ],
        (".yaml", ".yml"),
    ),
    "json": (
        [
            "SchemaStore/schemastore", "microsoft/vscode", "babel/babel",
            "prettier/prettier", "eslint/eslint", "webpack/webpack",
            "microsoft/TypeScript", "nlohmann/json", "jsonlint? no",
        ],
        (".json",),
    ),
    "toml": (
        [
            "rust-lang/cargo", "toml-lang/toml-test", "rust-lang/rust-analyzer",
            "bevyengine/bevy", "tokio-rs/tokio", "serde-rs/serde",
            "rust-lang/rustup", "denoland/deno", "astral-sh/uv",
            "pypa/hatch", "tauri-apps/tauri", "helix-editor/helix",
        ],
        (".toml",),
    ),
    "ini": (
        [
            "php/php-src", "benhoyt/inih", "python/mypy", "pytest-dev/pytest",
            "wine-mirror/wine", "systemd/systemd", "pypa/setuptools",
            "sqlalchemy/sqlalchemy", "pallets/flask", "django/django",
            "tox-dev/tox", "psf/requests",
        ],
        (".ini", ".cfg"),
    ),
    "xml": (
        [
            "apache/tomcat", "spring-projects/spring-framework",
            "mybatis/mybatis-3", "apache/maven", "android/architecture-samples",
            "apache/camel", "apache/ant", "eclipse-platform/eclipse.platform",
            "apache/struts", "AndroidX? no", "dbeaver/dbeaver",
        ],
        (".xml",),
    ),
    "swift": (
        [
            "Alamofire/Alamofire", "vapor/vapor", "apple/swift-nio",
            "ReactiveX/RxSwift", "apple/swift-package-manager",
            "pointfreeco/swift-composable-architecture", "SnapKit/SnapKit",
            "onevcat/Kingfisher", "kean/Nuke", "apple/swift-algorithms",
            "apple/swift-collections", "SwiftyJSON/SwiftyJSON",
            "realm/SwiftLint", "Quick/Quick", "Quick/Nimble",
        ],
        (".swift",),
    ),
    "cobol": (
        [
            "openmainframeproject/cobol-programming-course",
            "cicsdev/cics-banking-sample-application-cbsa",
            "shamrice/COBOL-Examples", "mapmeld/cobol-fizzbuzz",
            "azac/cobol-on-wheelchair",
            "olegkunitsyn/gnucobol-debug", "neopragma/cobol-unit-test",
            "openmainframeproject/cobol-check", "cicsdev/cics-genapp",
            "IBM/zopeneditor-sample", "aws-samples/aws-mainframe-modernization-carddemo",
            "mikebild? no", "rplig? no",
        ],
        (".cbl", ".cob", ".CBL", ".COB", ".cobol"),
    ),
}
# drop typo'd placeholders
for label, (repos, exts) in list(EXTRA_GITHUB.items()):
    EXTRA_GITHUB[label] = ([r for r in repos if "?" not in r], exts)


def main() -> int:
    out = Path(sys.argv[1])
    rng = random.Random(2)
    files_root = out / "files"
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    writer = bfc.CorpusWriter(files_root)
    import hashlib
    for split_dir in files_root.glob("*/"):
        for label_dir in split_dir.glob("*/"):
            for file in label_dir.iterdir():
                writer.seen.add(hashlib.sha1(file.read_bytes()).digest())
                writer.counts[(split_dir.name, label_dir.name)] += 1

    bfc.harvest_smol_xl(writer, None, scratch, rng)
    bfc.GITHUB_SOURCES = {**bfc.GITHUB_SOURCES, **EXTRA_GITHUB}
    import urllib.error
    for label, (repos, exts) in bfc.GITHUB_SOURCES.items():
        for repo in repos:
            try:
                bfc.GITHUB_SOURCES = {label: ([repo], exts)}
                bfc.harvest_github(writer, scratch)
            except Exception as err:  # tolerate 404s / truncated tars
                print(f"github {repo}: FAILED {err}", flush=True)
                (scratch / f"github.{repo.replace('/', '__')}.done").write_text("error\n")
    bfc.harvest_raw_files(writer, scratch)
    if not (scratch / "synthetic.done").exists():
        bfc.harvest_synthetic(writer, rng)
        (scratch / "synthetic.done").write_text("done\n")
    print(writer.summary())
    (out / "corpus_summary.txt").write_text(writer.summary() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

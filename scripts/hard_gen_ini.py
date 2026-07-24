"""Synthesize "hard" INI/config files for stress-testing a source-language classifier.

Every sample produced by synth_hard_ini() is unambiguously an INI-style config
file (.ini/.cfg: [sections], key=value pairs, ;/# comments), but each one is
deliberately laced with surface features that pull weak byte/token classifiers
(word units, punctuation runs, brackets from the first 4096 bytes) toward
markdown (prose comment paragraphs, bracketed multi-word headers that resemble
link references), batch (%VAR% expansions, drive-letter paths, URLs), or
python (True/None/False literals, comma lists, setup.cfg/tox.ini idioms).
Only the supplied random.Random instance is used, so output is reproducible.
"""

# ---------------------------------------------------------------------------
# Word pools (shared by all sub-styles; combined randomly, never verbatim).
# ---------------------------------------------------------------------------

_NOUNS = [
    "cache", "worker", "session", "proxy", "logging", "backend", "storage",
    "network", "database", "queue", "metrics", "auth", "token", "cluster",
    "replica", "handler", "socket", "buffer", "index", "archive", "mirror",
    "policy", "runtime", "sandbox", "monitor", "gateway", "schema", "bundle",
    "plugin", "release", "profile", "channel", "webhook", "ledger", "quota",
]

_ADJS = [
    "primary", "remote", "local", "shared", "default", "internal", "legacy",
    "secure", "verbose", "strict", "lazy", "global", "fallback", "staging",
    "nightly", "managed", "static", "dynamic", "rotating", "pinned",
    "temporary", "persistent", "external", "minimal", "extended", "trusted",
    "isolated", "custom", "upstream", "readonly", "batched", "async",
]

_KEY_SUFFIX = [
    "timeout", "limit", "path", "dir", "host", "port", "size", "level",
    "count", "mode", "url", "key", "file", "name", "interval", "retries",
    "threshold", "ttl", "prefix", "suffix", "backend", "driver", "user",
    "group", "target", "format", "depth", "window", "rate", "weight",
]

_PROSE = [
    "this", "section", "controls", "how", "the", "service", "handles",
    "incoming", "requests", "and", "when", "values", "are", "reloaded",
    "from", "disk", "note", "that", "changing", "these", "settings",
    "requires", "a", "restart", "of", "every", "running", "instance",
    "before", "they", "take", "effect", "see", "operators", "guide",
    "for", "details", "on", "tuning", "under", "heavy", "load", "most",
    "deployments", "should", "keep", "defaults", "unless", "profiling",
    "shows", "otherwise", "older", "releases", "ignored", "this", "field",
]

_HOSTS = [
    "example.com", "internal.corp", "api.example.org", "cdn.example.net",
    "mirror.example.io", "registry.local", "build.example.dev",
    "vault.example.co", "stats.example.app", "git.example.org",
]

_ENV_VARS = [
    "APPDATA", "PROGRAMFILES", "USERPROFILE", "TEMP", "SYSTEMROOT",
    "LOCALAPPDATA", "COMPUTERNAME", "PATH", "HOMEDRIVE", "WINDIR",
    "JAVA_HOME", "PYTHONPATH", "BUILD_DIR", "CONFIG_HOME", "CACHE_ROOT",
]

_UNIX_DIRS = [
    "usr/bin", "usr/local/bin", "opt/app", "var/lib", "var/log", "etc",
    "srv/data", "usr/share", "home/deploy", "run/lock",
]

_EXTS = ["py", "js", "toml", "yaml", "json", "md", "rst", "sh", "cfg", "txt"]

_BOOLS_PY = ["True", "False", "None"]
_BOOLS_INI = ["true", "false", "yes", "no", "on", "off", "1", "0"]


# ---------------------------------------------------------------------------
# Small helpers (all take rng explicitly).
# ---------------------------------------------------------------------------

def _word(rng):
    return rng.choice(rng.choice((_NOUNS, _ADJS)))


def _key(rng):
    style = rng.randrange(4)
    a = rng.choice(_ADJS)
    b = rng.choice(_NOUNS)
    c = rng.choice(_KEY_SUFFIX)
    if style == 0:
        return "%s_%s" % (b, c)
    if style == 1:
        return "%s_%s_%s" % (a, b, c)
    if style == 2:
        return "%s.%s" % (b, c)
    return "%s-%s" % (a, c)


def _sentence(rng, lo=6, hi=14):
    n = rng.randint(lo, hi)
    words = [rng.choice(_PROSE) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + rng.choice([".", ".", ".", "!", "?"])


def _number(rng):
    kind = rng.randrange(5)
    if kind == 0:
        return str(rng.randint(0, 9))
    if kind == 1:
        return str(rng.randint(10, 65535))
    if kind == 2:
        return "%.2f" % (rng.random() * rng.choice([1, 10, 100]))
    if kind == 3:
        return str(rng.choice([1024, 2048, 4096, 8192, 16384]) * rng.randint(1, 8))
    return "%d%s" % (rng.randint(1, 900), rng.choice(["ms", "s", "m", "h", "k", "M"]))


def _url(rng):
    scheme = rng.choice(["https", "http", "ftp", "ssh"])
    host = rng.choice(_HOSTS)
    depth = rng.randint(0, 3)
    parts = "/".join(_word(rng) for _ in range(depth))
    tail = "/" + parts if parts else ""
    if rng.random() < 0.3:
        tail += "?%s=%s" % (_word(rng), _number(rng))
    return "%s://%s%s" % (scheme, host, tail)


def _win_path(rng):
    drive = rng.choice(["C", "D", "E"])
    parts = [rng.choice(["Program Files", "Users", "ProgramData", "Windows", "Tools"])]
    parts += [_word(rng) for _ in range(rng.randint(1, 3))]
    leaf = "%s.%s" % (_word(rng), rng.choice(["exe", "bat", "dll", "ini", "log"]))
    return drive + ":\\" + "\\".join(parts) + "\\" + leaf


def _unix_path(rng):
    base = rng.choice(_UNIX_DIRS)
    parts = [_word(rng) for _ in range(rng.randint(0, 2))]
    leaf = _word(rng) + rng.choice(["", ".%s" % rng.choice(_EXTS)])
    return "/" + "/".join([base] + parts + [leaf])


def _comment_block(rng, mark, lo=1, hi=4):
    lines = []
    for _ in range(rng.randint(lo, hi)):
        lines.append("%s %s" % (mark, _sentence(rng)))
    return lines


# ---------------------------------------------------------------------------
# Sub-styles.  Each returns a list of text lines.
# ---------------------------------------------------------------------------

def _style_prose_comments(rng):
    """INI dominated by long ;/# comment paragraphs, like markdown prose."""
    mark = rng.choice([";", "#"])
    lines = _comment_block(rng, mark, 2, 5)
    lines.append("")
    for _ in range(rng.randint(2, 4)):
        lines.append("[%s]" % _key(rng).replace(".", "_"))
        if rng.random() < 0.9:
            lines.extend(_comment_block(rng, mark, 1, 4))
        for _ in range(rng.randint(2, 5)):
            val = rng.choice([_number(rng), rng.choice(_BOOLS_INI), _word(rng)])
            lines.append("%s = %s" % (_key(rng), val))
            if rng.random() < 0.4:
                lines.append("%s %s" % (mark, _sentence(rng, 4, 9)))
        lines.append("")
        if rng.random() < 0.5:
            lines.extend(_comment_block(rng, mark, 1, 3))
            lines.append("")
    return lines


def _style_spacey_sections(rng):
    """Multi-word [section names with spaces] that resemble md link refs."""
    lines = []
    if rng.random() < 0.6:
        lines.append("; %s" % _sentence(rng, 4, 8))
        lines.append("")
    for _ in range(rng.randint(3, 6)):
        n = rng.randint(2, 4)
        title = " ".join(_word(rng) for _ in range(n))
        if rng.random() < 0.4:
            title = title.title()
        lines.append("[%s]" % title)
        for _ in range(rng.randint(2, 5)):
            pick = rng.random()
            if pick < 0.3:
                val = _url(rng)
            elif pick < 0.6:
                val = "%s: %s" % (_word(rng), _sentence(rng, 3, 6))
            else:
                val = _number(rng)
            lines.append("%s = %s" % (_key(rng), val))
        lines.append("")
    return lines


def _style_batchlike(rng):
    """Values full of %VAR%, $ENV, drive letters, and URLs (batch-flavored)."""
    lines = ["; Generated for host %s" % rng.choice(_HOSTS).split(".")[0]]
    for _ in range(rng.randint(2, 4)):
        lines.append("")
        lines.append("[%s]" % _key(rng))
        for _ in range(rng.randint(3, 6)):
            pick = rng.random()
            if pick < 0.25:
                val = "%%%s%%\\%s" % (rng.choice(_ENV_VARS), _word(rng))
            elif pick < 0.45:
                val = "$%s/%s" % (rng.choice(_ENV_VARS), _word(rng))
            elif pick < 0.65:
                val = _win_path(rng)
            elif pick < 0.85:
                val = _url(rng)
            else:
                val = "%s.%s %s" % (_word(rng), rng.choice(["exe", "bat", "cmd"]),
                                    rng.choice(["/S", "/Q", "-y", "--force", "/norestart"]))
            key = _key(rng).replace(".", "_").replace("-", "_")
            eq = rng.choice(["=", " = "])
            lines.append("%s%s%s" % (key, eq, val))
    lines.append("")
    return lines


def _style_systemd(rng):
    """systemd unit file: [Unit]/[Service]/[Install] with Exec lines."""
    app = "%s-%s" % (rng.choice(_ADJS), rng.choice(_NOUNS))
    lines = ["[Unit]"]
    lines.append("Description=%s %s daemon" % (rng.choice(_ADJS).capitalize(),
                                               rng.choice(_NOUNS)))
    if rng.random() < 0.7:
        lines.append("After=network%s.target" % rng.choice(["", "-online"]))
    if rng.random() < 0.4:
        lines.append("Wants=%s.service" % rng.choice(_NOUNS))
    if rng.random() < 0.4:
        lines.append("Documentation=%s" % _url(rng))
    lines.append("")
    lines.append("[Service]")
    lines.append("Type=%s" % rng.choice(["simple", "forking", "notify", "oneshot"]))
    flag = rng.choice(["--config", "--listen", "--workers", "--data-dir"])
    lines.append("ExecStart=/%s/%s %s %s" % (rng.choice(["usr/bin", "usr/local/bin", "opt/%s/bin" % app]),
                                             app, flag,
                                             rng.choice([_unix_path(rng), _number(rng)])))
    if rng.random() < 0.5:
        lines.append("ExecReload=/bin/kill -HUP $MAINPID")
    for k, v in (("Restart", rng.choice(["always", "on-failure", "no"])),
                 ("RestartSec", _number(rng)),
                 ("User", rng.choice(["root", "daemon", app.split("-")[1], "nobody"])),
                 ("Environment", "%s=%s" % (rng.choice(_ENV_VARS), _unix_path(rng))),
                 ("WorkingDirectory", _unix_path(rng)),
                 ("LimitNOFILE", str(rng.choice([1024, 4096, 65536])))):
        if rng.random() < 0.65:
            lines.append("%s=%s" % (k, v))
    lines.append("")
    lines.append("[Install]")
    lines.append("WantedBy=%s.target" % rng.choice(["multi-user", "default", "graphical"]))
    lines.append("")
    return lines


def _style_editorconfig(rng):
    """editorconfig: glob section headers like [*.py] and style knobs."""
    lines = []
    if rng.random() < 0.7:
        lines.append("# %s" % _sentence(rng, 4, 8))
    lines.append("root = %s" % rng.choice(["true", "false"]))
    lines.append("")
    lines.append("[*]")
    for k, v in (("charset", rng.choice(["utf-8", "latin1", "utf-16"])),
                 ("end_of_line", rng.choice(["lf", "crlf"])),
                 ("insert_final_newline", rng.choice(["true", "false"])),
                 ("trim_trailing_whitespace", rng.choice(["true", "false"])),
                 ("indent_style", rng.choice(["space", "tab"]))):
        if rng.random() < 0.8:
            lines.append("%s = %s" % (k, v))
    exts = rng.sample(_EXTS, rng.randint(2, 4))
    for ext in exts:
        lines.append("")
        header = rng.choice(["[*.%s]" % ext,
                             "[{*.%s,*.%s}]" % (ext, rng.choice(_EXTS)),
                             "[%s/**.%s]" % (_word(rng), ext)])
        lines.append(header)
        lines.append("indent_size = %d" % rng.choice([2, 3, 4, 8]))
        if rng.random() < 0.5:
            lines.append("max_line_length = %d" % rng.choice([79, 88, 100, 120]))
        if rng.random() < 0.3:
            lines.append("# %s" % _sentence(rng, 3, 7))
    lines.append("")
    return lines


def _style_gitconfig(rng):
    """git-config: [section "subsection"] with tab-indented key = value."""
    indent = rng.choice(["\t", "    ", "  "])
    lines = []
    sections = rng.randint(3, 5)
    for _ in range(sections):
        top = rng.choice(["core", "remote", "branch", "alias", "user", "http",
                          "submodule", "credential", "diff", "fetch"])
        if top in ("remote", "branch", "submodule") or rng.random() < 0.3:
            lines.append('[%s "%s"]' % (top, _word(rng)))
        else:
            lines.append("[%s]" % top)
        for _ in range(rng.randint(2, 4)):
            pick = rng.random()
            if pick < 0.3:
                val = _url(rng)
            elif pick < 0.5:
                val = _unix_path(rng)
            elif pick < 0.7:
                val = rng.choice(_BOOLS_INI[:4])
            else:
                val = _number(rng)
            k = rng.choice(_KEY_SUFFIX) + rng.choice(["", rng.choice(_NOUNS).capitalize()])
            lines.append("%s%s = %s" % (indent, k, val))
        if rng.random() < 0.3:
            lines.append("%s# %s" % (indent, _sentence(rng, 3, 7)))
        lines.append("")
    return lines


def _style_pythonic(rng):
    """setup.cfg / tox.ini / pytest.ini flavor: True/None, comma lists,
    multi-line indented list values."""
    lines = []
    kind = rng.randrange(3)
    if kind == 0:
        lines.append("[metadata]")
        lines.append("name = %s-%s" % (rng.choice(_ADJS), rng.choice(_NOUNS)))
        lines.append("version = %d.%d.%d" % (rng.randint(0, 4), rng.randint(0, 12),
                                             rng.randint(0, 30)))
        if rng.random() < 0.6:
            lines.append("description = %s" % _sentence(rng, 4, 9)[:-1])
        lines.append("")
        lines.append("[options]")
        lines.append("zip_safe = %s" % rng.choice(_BOOLS_PY[:2]))
        lines.append("include_package_data = %s" % rng.choice(_BOOLS_PY[:2]))
        lines.append("install_requires =")
        for _ in range(rng.randint(2, 5)):
            lines.append("    %s>=%d.%d" % (rng.choice(_NOUNS), rng.randint(0, 5),
                                            rng.randint(0, 20)))
    elif kind == 1:
        envs = ["py3%d" % rng.randint(8, 13) for _ in range(rng.randint(2, 4))]
        lines.append("[tox]")
        lines.append("envlist = %s" % ", ".join(envs + ["lint"]))
        lines.append("skipsdist = %s" % rng.choice(_BOOLS_PY[:2]))
        lines.append("")
        lines.append("[testenv]")
        lines.append("deps =")
        for _ in range(rng.randint(2, 4)):
            lines.append("    %s" % rng.choice(_NOUNS))
        lines.append("commands = pytest {posargs} -%s" % rng.choice(["q", "x", "v"]))
        lines.append("setenv =")
        lines.append("    %s = %s" % (rng.choice(_ENV_VARS), rng.choice(_BOOLS_PY)))
    else:
        lines.append("[pytest]")
        lines.append("addopts = %s" % " ".join(
            rng.sample(["-ra", "--strict-markers", "-q", "--tb=short",
                        "--maxfail=%d" % rng.randint(1, 5)], rng.randint(2, 3))))
        lines.append("testpaths = %s" % ", ".join(
            rng.sample(["tests", "unit", "integration", "functional"], rng.randint(1, 3))))
        lines.append("python_files = test_*.py")
        lines.append("filterwarnings =")
        lines.append("    %s" % rng.choice(["error", "ignore::DeprecationWarning",
                                            "default::UserWarning"]))
    lines.append("")
    lines.append("[%s]" % rng.choice(["flake8", "mypy", "isort", "coverage:run", "bdist_wheel"]))
    for _ in range(rng.randint(2, 4)):
        pick = rng.random()
        if pick < 0.35:
            val = rng.choice(_BOOLS_PY)
        elif pick < 0.7:
            val = ", ".join(_word(rng) for _ in range(rng.randint(2, 4)))
        else:
            val = _number(rng)
        lines.append("%s = %s" % (_key(rng).replace("-", "_").replace(".", "_"), val))
    lines.append("")
    return lines


_STYLES = [
    _style_prose_comments,
    _style_spacey_sections,
    _style_batchlike,
    _style_systemd,
    _style_editorconfig,
    _style_gitconfig,
    _style_pythonic,
]


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def synth_hard_ini(rng):
    """Return one hard-but-genuine INI/config file as ASCII bytes (120B..~3KB)."""
    style = rng.choice(_STYLES)
    lines = style(rng)

    # Grow small samples by appending extra sections from any style.
    text = "\n".join(lines)
    while len(text) < 120 or (len(text) < 400 and rng.random() < 0.5):
        extra = rng.choice(_STYLES)(rng)
        text = text.rstrip("\n") + "\n\n" + "\n".join(extra)

    # Cap at ~3KB on a line boundary so we never cut a line in half.
    if len(text) > 3000:
        kept = []
        total = 0
        for ln in text.split("\n"):
            if total + len(ln) + 1 > 3000:
                break
            kept.append(ln)
            total += len(ln) + 1
        text = "\n".join(kept)

    if not text.endswith("\n"):
        text += "\n"
    return text.encode("ascii")


if __name__ == "__main__":
    import random
    _rng = random.Random(0)
    for _i in range(3):
        print(synth_hard_ini(_rng).decode("ascii"))
        print("=" * 40)

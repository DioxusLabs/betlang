"""Synthesizers for "hard boundary" JS/TS training samples.

This module produces source files that sit near the decision boundary of a
JavaScript-vs-TypeScript classifier (a small Magika-style model that sees the
first 4096 bytes as case-folded word tokens, punctuation runs and brackets).

- synth_hard_javascript(rng) -> bytes: files that ARE plain JavaScript but
  look "typed" (JSDoc annotations, .d.ts-flavored naming, async-heavy ESM,
  CommonJS with destructuring/default params).
- synth_hard_typescript(rng) -> bytes: files that ARE TypeScript but are
  almost entirely plain JS (sparse type touches, no imports, config-style
  default exports, a barely-typed class).

All randomness flows through the provided random.Random instance so output
is reproducible. Stdlib only, ASCII only.
"""

# ---------------------------------------------------------------------------
# Word pools and small helpers
# ---------------------------------------------------------------------------

_NOUNS = [
    "user", "cache", "token", "session", "record", "widget", "payload",
    "bucket", "worker", "queue", "batch", "route", "config", "schema",
    "metric", "report", "invoice", "order", "product", "account", "profile",
    "cluster", "shard", "cursor", "stream", "buffer", "handle", "policy",
    "ticket", "vendor", "region", "channel", "manifest", "snapshot",
    "registry", "catalog", "segment", "quota", "license", "webhook",
]

_VERBS = [
    "load", "save", "fetch", "parse", "build", "merge", "split", "filter",
    "resolve", "reject", "encode", "decode", "flush", "sync", "track",
    "apply", "reduce", "collect", "emit", "drain", "hydrate", "validate",
    "normalize", "serialize", "restore", "publish", "consume", "rotate",
    "prune", "index", "attach", "detach", "compare", "clone", "format",
]

_ADJS = [
    "active", "stale", "pending", "remote", "local", "shared", "primary",
    "backup", "hidden", "compact", "internal", "legacy", "draft", "final",
    "partial", "nested", "cached", "sorted", "unique", "verbose",
]

_STR_WORDS = [
    "alpha", "beta", "gamma", "delta", "omega", "north", "south", "east",
    "west", "ready", "closed", "open", "retry", "fatal", "warn", "info",
    "debug", "trace", "green", "amber", "coral", "ivory", "slate", "onyx",
]

_COMMENT_BITS = [
    "keep this in sync with the server side",
    "tuned by hand, do not change casually",
    "falls back to defaults when the flag is off",
    "used by the nightly batch job",
    "cheap enough to run on every request",
    "see the runbook for the failure modes",
    "order matters here",
    "this mirrors the upstream behaviour",
    "avoids an extra allocation in the hot path",
    "intentionally permissive",
    "the retry budget is shared across workers",
    "measured on the staging cluster",
]


def _cap(w):
    return w[:1].upper() + w[1:]


def _camel(rng, parts=2):
    ws = [rng.choice(_VERBS)] + [rng.choice(_NOUNS) for _ in range(parts - 1)]
    return ws[0] + "".join(_cap(w) for w in ws[1:])


def _noun_camel(rng, parts=2):
    ws = [rng.choice(_ADJS if rng.random() < 0.4 else _NOUNS)]
    ws += [rng.choice(_NOUNS) for _ in range(parts - 1)]
    return ws[0] + "".join(_cap(w) for w in ws[1:])


def _pascal(rng, parts=2):
    return "".join(_cap(rng.choice(_NOUNS)) for _ in range(parts))


def _const_name(rng):
    return "_".join(rng.choice(_NOUNS).upper() for _ in range(rng.randint(1, 2)))


def _num(rng):
    kind = rng.random()
    if kind < 0.3:
        return str(rng.randint(0, 9))
    if kind < 0.6:
        return str(rng.choice([16, 32, 64, 100, 128, 250, 256, 500, 1000, 1024, 3600, 8080]))
    if kind < 0.8:
        return str(rng.randint(10, 99999))
    return "%d.%d" % (rng.randint(0, 99), rng.randint(1, 99))


def _str_val(rng):
    kind = rng.random()
    if kind < 0.35:
        return rng.choice(_STR_WORDS)
    if kind < 0.55:
        return rng.choice(_STR_WORDS) + "-" + rng.choice(_STR_WORDS)
    if kind < 0.75:
        return "/".join(["api", rng.choice(_NOUNS) + "s", rng.choice(_STR_WORDS)])
    if kind < 0.9:
        return rng.choice(_NOUNS) + "." + rng.choice(["json", "yaml", "txt", "log", "csv"])
    return "https://" + rng.choice(_STR_WORDS) + ".example.com/" + rng.choice(_NOUNS)


def _line_comment(rng):
    return "// " + rng.choice(_COMMENT_BITS)


def _maybe_comment(rng, p=0.4):
    return _line_comment(rng) + "\n" if rng.random() < p else ""


def _target_len(rng):
    base = rng.choice([220, 380, 600, 900, 1300, 1900, 2600, 3300])
    return base + rng.randint(-40, 160)


def _uniq(rng, maker, used):
    """Draw a fresh identifier not previously used in this file."""
    name = maker(rng)
    for _ in range(8):
        if name not in used:
            break
        name = maker(rng)
    if name in used:
        name += str(rng.randint(2, 99))
    used.add(name)
    return name


def _distinct_pair(rng, maker):
    a = maker(rng)
    b = maker(rng)
    while b == a:
        b = maker(rng)
    return a, b


def _body_stmts(rng, indent="  "):
    """A few plain-JS statements usable inside any function body."""
    out = []
    n = rng.randint(1, 3)
    declared = set()
    for _ in range(n):
        k = rng.random()
        a = _uniq(rng, _noun_camel, declared)
        b = _noun_camel(rng)
        while b == a:
            b = _noun_camel(rng)
        if k < 0.25:
            out.append('%sconst %s = %s ?? "%s";' % (indent, a, b, _str_val(rng)))
        elif k < 0.5:
            out.append("%sif (!%s) {\n%s  return %s;\n%s}" % (
                indent, a, indent, rng.choice(["null", "[]", "false", "undefined"]), indent))
        elif k < 0.75:
            out.append("%sconst %s = %s.map((item) => item.%s).filter(Boolean);" % (
                indent, a, b + "s", rng.choice(_NOUNS)))
        else:
            out.append("%slet %s = %s;" % (indent, a, _num(rng)))
    return "\n".join(out)


def _assemble(rng, header, block_fn, target):
    parts = [header] if header else []
    size = len(header)
    guard = 0
    while (size < target or size < 150) and guard < 60:
        block = block_fn(rng)
        parts.append(block)
        size += len(block) + 1
        guard += 1
    text = "\n".join(p for p in parts if p) + "\n"
    if len(text) > 4000:
        cut = text[:4000]
        idx = cut.rfind("\n}")
        if idx > 150:
            text = cut[: idx + 2] + "\n"
    return text


# ---------------------------------------------------------------------------
# Hard JavaScript sub-styles (valid JS that smells like TS)
# ---------------------------------------------------------------------------

def _js_jsdoc_module(rng):
    """ESM classes and functions saturated with JSDoc type annotations."""
    ty = _pascal(rng)
    mod = rng.choice(["node:crypto", "node:path", "node:url", "./" + rng.choice(_NOUNS) + ".js"])
    imp = rng.choice(["createHash", "resolve", "fileURLToPath", _camel(rng)])
    header = 'import { %s } from "%s";\n' % (imp, mod)
    header += "\n/**\n * @typedef {Object} %s\n" % ty
    for _ in range(rng.randint(2, 4)):
        header += " * @property {%s} %s\n" % (
            rng.choice(["string", "number", "boolean", "string[]", "number|null"]),
            _noun_camel(rng, 1))
    header += " */\n\n/** @type {Map<string, %s>} */\nconst %s = new Map();\n" % (
        ty, _noun_camel(rng))
    used = {ty}

    def block(rng):
        k = rng.random()
        fn = _uniq(rng, _camel, used)
        p1, p2 = _distinct_pair(rng, lambda r: _noun_camel(r, 1))
        if k < 0.45:
            doc = "/**\n * @param {string} %s\n * @param {number} [%s]\n * @returns {%s|undefined}\n */" % (p1, p2, ty)
            return "%s\n%sexport function %s(%s, %s = %s) {\n%s\n  return undefined;\n}\n" % (
                doc, _maybe_comment(rng), fn, p1, p2, _num(rng), _body_stmts(rng))
        if k < 0.75:
            cls = _uniq(rng, _pascal, used)
            f1 = _noun_camel(rng, 1)
            return (
                "export class %s {\n"
                "  /** @param {%s} options */\n"
                "  constructor(options) {\n"
                "    /** @type {%s} */\n"
                "    this.%s = options;\n"
                "    /** @type {number} */\n"
                "    this.%s = %s;\n"
                "  }\n\n"
                "  /** @returns {boolean} */\n"
                "  %s() {\n%s\n    return true;\n  }\n"
                "}\n" % (cls, ty, ty, f1, _noun_camel(rng, 1), _num(rng),
                          _camel(rng), _body_stmts(rng, "    "))
            )
        cn = _uniq(rng, _const_name, used)
        return "/** @type {readonly string[]} */\nconst %s = [%s];\n" % (
            cn, ", ".join('"%s"' % _str_val(rng) for _ in range(rng.randint(2, 4))))

    return _assemble(rng, header, block, _target_len(rng))


def _js_dts_flavored(rng):
    """Plain JS with declaration-file flavored naming (IThing, TResult, enums)."""
    kind = _pascal(rng, 1) + "Kind"
    members = [_cap(rng.choice(_STR_WORDS)) for _ in range(rng.randint(3, 5))]
    header = _maybe_comment(rng, 0.6)
    header += "export const %s = Object.freeze({\n" % kind
    header += "".join("  %s: %d,\n" % (m, i) for i, m in enumerate(dict.fromkeys(members)))
    header += "});\n"
    used = {kind}

    def block(rng):
        k = rng.random()
        iface = _uniq(rng, lambda r: "I" + _pascal(r), used)
        if k < 0.35:
            return (
                "%sexport class %s {\n"
                "  constructor(kind, value) {\n"
                "    this.kind = kind;\n"
                "    this.value = value;\n"
                "  }\n\n"
                "  %s() {\n    return this.kind === %s.%s;\n  }\n"
                "}\n" % (_maybe_comment(rng), iface, _camel(rng), kind,
                          rng.choice(members))
            )
        if k < 0.65:
            tname = _uniq(rng, lambda r: "T" + _pascal(r, 1), used)
            return (
                "export function is%s(value) {\n"
                '  return value != null && typeof value.%s === "%s";\n'
                "}\n\n"
                "export function assert%s(value) {\n"
                "  if (!is%s(value)) {\n"
                '    throw new TypeError("expected %s");\n'
                "  }\n  return value;\n}\n" % (
                    tname, _noun_camel(rng, 1),
                    rng.choice(["string", "number", "function"]),
                    tname, tname, tname.lower())
            )
        rec = _uniq(rng, lambda r: _pascal(r) + "Map", used)
        return "export const %s = {\n%s\n};\n" % (
            rec,
            "\n".join('  %s: "%s",' % (_noun_camel(rng, 1), _str_val(rng))
                      for _ in range(rng.randint(2, 4))))

    return _assemble(rng, header, block, _target_len(rng))


def _js_async_heavy(rng):
    """ESM module dominated by async/await, promises and async generators."""
    base = _noun_camel(rng)
    header = _maybe_comment(rng, 0.5)
    header += 'const %s = "%s";\nconst %s = %s;\n' % (
        base + "Url", _str_val(rng), _noun_camel(rng, 1) + "Limit", _num(rng))
    used = {base + "Url"}

    def block(rng):
        k = rng.random()
        fn = _uniq(rng, _camel, used)
        if k < 0.35:
            return (
                "export async function %s(id, signal) {\n"
                "  const res = await fetch(`${%s}/${id}`, { signal });\n"
                "  if (!res.ok) {\n"
                "    throw new Error(`%s failed: ${res.status}`);\n"
                "  }\n"
                "  const data = await res.json();\n%s\n"
                "  return data;\n"
                "}\n" % (fn, base + "Url", fn, _body_stmts(rng))
            )
        if k < 0.6:
            return (
                "export async function* %s(ids) {\n"
                "  for (const id of ids) {\n"
                "    %s\n"
                "    yield await %s(id);\n"
                "  }\n"
                "}\n" % (fn, _line_comment(rng), _camel(rng))
            )
        if k < 0.85:
            a, b = _distinct_pair(rng, _noun_camel)
            return (
                "export async function %s(items) {\n"
                "  const %s = await Promise.all(items.map(async (item) => {\n"
                "    const %s = await item.%s();\n"
                "    return %s ?? %s;\n"
                "  }));\n"
                "  return %s.filter((x) => x !== null);\n"
                "}\n" % (fn, a, b, _camel(rng), b, _num(rng), a)
            )
        return (
            "async function %s(ms) {\n"
            "  await new Promise((resolve) => setTimeout(resolve, ms));\n"
            "}\n" % fn
        )

    return _assemble(rng, header, block, _target_len(rng))


def _js_commonjs(rng):
    """CommonJS with destructured requires, default params, module.exports."""
    mods = rng.sample(["fs/promises", "path", "os", "util", "events", "url"], 2)
    header = '"use strict";\n\n' if rng.random() < 0.7 else ""
    r1, r2 = _distinct_pair(rng, _camel)
    header += 'const { %s, %s } = require("%s");\n' % (r1, r2, mods[0])
    header += 'const %s = require("%s");\n' % (mods[1].split("/")[0], mods[1])
    exported = []
    used = {r1, r2}

    def block(rng):
        k = rng.random()
        fn = _uniq(rng, _camel, used)
        exported.append(fn)
        p1, p2 = _distinct_pair(rng, lambda r: _noun_camel(r, 1))
        if k < 0.4:
            d1 = _uniq(rng, lambda r: _noun_camel(r, 1), {p1, p2})
            d2 = _uniq(rng, lambda r: _noun_camel(r, 1), {p1, p2, d1})
            return (
                '%sfunction %s(%s, %s = %s, prefix = "%s") {\n'
                "  const { %s, %s = [] } = %s;\n%s\n"
                "  return { %s, prefix };\n"
                "}\n" % (_maybe_comment(rng), fn, p1, p2, _num(rng), _str_val(rng),
                          d1, d2, p1, _body_stmts(rng), p2)
            )
        if k < 0.7:
            return (
                "function %s({ %s = %s, %s = \"%s\" } = {}) {\n%s\n"
                "  return %s;\n"
                "}\n" % (fn, p1, _num(rng), p2, _str_val(rng),
                          _body_stmts(rng), p1)
            )
        exported.pop()
        cn = _uniq(rng, _const_name, used)
        exported.append(cn)
        return "const %s = %s; %s\n" % (cn, _num(rng), _line_comment(rng))

    text = _assemble(rng, header, block, _target_len(rng))
    names = list(dict.fromkeys(exported))[:6]
    text += "\nmodule.exports = { %s };\n" % ", ".join(names)
    return text


# ---------------------------------------------------------------------------
# Hard TypeScript sub-styles (valid TS that is ~95 percent plain JS)
# ---------------------------------------------------------------------------

def _ts_sparse_types(rng):
    """Mostly plain JS; exactly a couple of tiny type touches sprinkled in."""
    touches = rng.sample(["param", "interface", "asconst", "generic", "satisfies"], 2)
    header = _maybe_comment(rng, 0.5)
    if "interface" in touches:
        iface = _pascal(rng)
        header += "interface %s {\n  %s: string;\n  %s: number;\n}\n\n" % (
            iface, _noun_camel(rng, 1), _noun_camel(rng, 1))
    else:
        iface = None

    used = {"param": False, "generic": False, "asconst": False,
            "satisfies": False, "iface": False}
    names = {iface} if iface else set()

    def block(rng):
        k = rng.random()
        fn = _uniq(rng, _camel, names)
        p1 = _noun_camel(rng, 1)
        if "param" in touches and not used["param"]:
            used["param"] = True
            return (
                "export function %s(%s: string, options = {}) {\n%s\n"
                "  return %s.trim();\n"
                "}\n" % (fn, p1, _body_stmts(rng), p1)
            )
        if "generic" in touches and not used["generic"]:
            used["generic"] = True
            arr = _uniq(rng, lambda r: _noun_camel(r, 1) + "s", names)
            return (
                "const %s: Array<number> = [];\n\n"
                "export function %s(value) {\n"
                "  %s.push(value | 0);\n"
                "  return %s.length;\n"
                "}\n" % (arr, fn, arr, arr)
            )
        if "asconst" in touches and not used["asconst"]:
            used["asconst"] = True
            return "const %s = [%s] as const;\n" % (
                _uniq(rng, _const_name, names),
                ", ".join('"%s"' % _str_val(rng) for _ in range(rng.randint(2, 4))))
        if "satisfies" in touches and not used["satisfies"]:
            used["satisfies"] = True
            k1, k2 = _distinct_pair(rng, lambda r: _noun_camel(r, 1))
            return (
                "export const %s = {\n  %s: %s,\n  %s: \"%s\",\n} satisfies Record<string, unknown>;\n" % (
                    _uniq(rng, _noun_camel, names), k1, _num(rng),
                    k2, _str_val(rng))
            )
        if iface and not used["iface"]:
            used["iface"] = True
            return (
                "export function %s(entry: %s) {\n"
                "  return `${entry.%s}`;\n"
                "}\n" % (fn, iface, _noun_camel(rng, 1))
            )
        if k < 0.5:
            return (
                "%sexport function %s(%s, limit = %s) {\n%s\n"
                "  return %s;\n"
                "}\n" % (_maybe_comment(rng), fn, p1, _num(rng),
                          _body_stmts(rng), p1)
            )
        return (
            "function %s(list) {\n"
            "  return list.reduce((acc, item) => acc + (item.%s || 0), 0);\n"
            "}\n" % (fn, _noun_camel(rng, 1))
        )

    return _assemble(rng, header, block, _target_len(rng))


def _ts_no_imports(rng):
    """Standalone TS: no imports at all, a lone type alias, plain helpers."""
    alias = _pascal(rng, 1) + "Id"
    header = _maybe_comment(rng, 0.5)
    header += "type %s = string;\n" % alias
    used = {alias}

    def block(rng):
        k = rng.random()
        fn = _uniq(rng, _camel, used)
        p1 = _noun_camel(rng, 1)
        if k < 0.3:
            return (
                "export function %s(id: %s) {\n"
                '  return id.startsWith("%s") ? id.slice(%s) : id;\n'
                "}\n" % (fn, alias, rng.choice(_STR_WORDS), _num(rng))
            )
        if k < 0.6:
            return (
                "%sexport function %s(%s, fallback = %s) {\n%s\n"
                "  return %s ?? fallback;\n"
                "}\n" % (_maybe_comment(rng), fn, p1, _num(rng),
                          _body_stmts(rng), p1)
            )
        if k < 0.8:
            return (
                "export const %s = (a, b) => {\n"
                "  return a.%s === b.%s ? 0 : a.%s < b.%s ? -1 : 1;\n"
                "};\n" % ((fn,) + (_noun_camel(rng, 1),) * 4)
            )
        return "const %s = %s; %s\n" % (
            _uniq(rng, _const_name, used), _num(rng), _line_comment(rng))

    return _assemble(rng, header, block, _target_len(rng))


def _ts_config_default(rng):
    """Config-style TS: one big `export default { ... }` object literal."""
    header = _maybe_comment(rng, 0.7)
    tail = rng.random()
    lines = [header + "export default {"]
    n = rng.randint(4, 9)
    seen = set()
    for _ in range(n):
        key = _uniq(rng, lambda r: _noun_camel(r, 1), seen)
        k = rng.random()
        if k < 0.3:
            lines.append('  %s: "%s",' % (key, _str_val(rng)))
        elif k < 0.5:
            lines.append("  %s: %s," % (key, _num(rng)))
        elif k < 0.65:
            lines.append("  %s: %s," % (key, rng.choice(["true", "false"])))
        elif k < 0.85:
            inner = ", ".join('"%s"' % _str_val(rng) for _ in range(rng.randint(2, 3)))
            lines.append("  %s: [%s]," % (key, inner))
        else:
            lines.append("  %s: {" % key)
            inner_seen = set()
            for _ in range(rng.randint(2, 3)):
                ikey = _uniq(rng, lambda r: _noun_camel(r, 1), inner_seen)
                lines.append('    %s: "%s",' % (ikey, _str_val(rng)))
            lines.append("    " + _line_comment(rng))
            lines.append("  },")
    if tail < 0.4:
        lines.append("} satisfies Record<string, unknown>;")
    elif tail < 0.7:
        lines.append("} as const;")
    else:
        lines[0] = header + "const config: { [key: string]: unknown } = {"
        lines.append("};\n\nexport default config;")
    text = "\n".join(lines) + "\n"
    while len(text) < 150:
        text += "\n" + _line_comment(rng) + "\n"
    return text


def _ts_barely_typed_class(rng):
    """A plain-looking class with a single typed field or return annotation."""
    cls = _pascal(rng)
    field = _noun_camel(rng, 1)
    header = _maybe_comment(rng, 0.5)
    header += (
        "export class %s {\n"
        "  private %s: Map<string, number> = new Map();\n"
        "  options;\n\n"
        "  constructor(options = {}) {\n"
        "    this.options = { ...options };\n"
        "  }\n" % (cls, field)
    )
    used = {field, "options"}

    def block(rng):
        k = rng.random()
        m = _uniq(rng, _camel, used)
        p1 = _noun_camel(rng, 1)
        if k < 0.3:
            return (
                "\n  %s(%s): void {\n"
                "    this.%s.set(%s, (this.%s.get(%s) || 0) + 1);\n"
                "  }\n" % (m, p1, field, p1, field, p1)
            )
        if k < 0.6:
            return (
                "\n  %s(%s, fallback = %s) {\n%s\n"
                "    return this.%s.get(%s) ?? fallback;\n"
                "  }\n" % (m, p1, _num(rng), _body_stmts(rng, "    "),
                            field, p1)
            )
        if k < 0.8:
            return (
                "\n  get %s() {\n    return this.%s.size;\n  }\n" % (
                    _uniq(rng, lambda r: _noun_camel(r, 1), used), field)
            )
        return (
            "\n  %s\n  %s() {\n    this.%s.clear();\n  }\n" % (
                _line_comment(rng), m, field)
        )

    text = _assemble(rng, header, block, max(150, _target_len(rng) - 120))
    text = text.rstrip("\n") + "\n}\n"
    helper = _camel(rng)
    text += "\nexport function %s() {\n  return new %s();\n}\n" % (helper, cls)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synth_hard_javascript(rng):
    """Return bytes of a plain-JavaScript file that leans toward TypeScript."""
    style = rng.choice([
        _js_jsdoc_module,
        _js_dts_flavored,
        _js_async_heavy,
        _js_commonjs,
    ])
    return style(rng).encode("ascii")


def synth_hard_typescript(rng):
    """Return bytes of a TypeScript file that leans toward plain JavaScript."""
    style = rng.choice([
        _ts_sparse_types,
        _ts_no_imports,
        _ts_config_default,
        _ts_barely_typed_class,
    ])
    return style(rng).encode("ascii")


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    sep = "=" * 40
    for label, fn in (("hard JavaScript", synth_hard_javascript),
                      ("hard TypeScript", synth_hard_typescript)):
        for i in range(2):
            data = fn(rng)
            print(sep)
            print("%s sample %d (%d bytes)" % (label, i + 1, len(data)))
            print(sep)
            print(data.decode("ascii"))

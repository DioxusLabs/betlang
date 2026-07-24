"""Synthesize "hard boundary" training files for a C-vs-C++ classifier.

The target model (a small Magika distillation) sees only the first 4096
bytes of a file, tokenized into case-folded words, punctuation runs and
brackets.  It tends to confuse C and C++, so this module emits samples
that sit deliberately close to the decision boundary while remaining
unambiguous to a human labeler:

  synth_hard_c(rng)   -> bytes of genuinely-C code (.c) that borrows C++
      surface features: vtable-style OO structs, class-like containers
      with new_*/delete_* functions, modern C with // comments and bool,
      macro pseudo-templates, and opaque-handle headers.

  synth_hard_cpp(rng) -> bytes of genuinely-C++ code (.cpp) that stays
      C-flavored (procedural printf/malloc-style code, no heavy STL soup)
      but always carries 2-4 unambiguous C++ markers near the top of the
      file: <cstdio>/<vector>-style includes, a real class with ctor/dtor
      and public:/private: sections, std:: usage, new/delete allocations,
      template <typename T> helpers, or a single std::cout line.

All randomness comes from the caller-provided random.Random instance so
generation is reproducible.  Output is ASCII, roughly 150 B to 4 KB.
"""

import random  # used only for the type of `rng` and the __main__ demo


# ---------------------------------------------------------------------------
# word pools (identifier fragments)
# ---------------------------------------------------------------------------

NOUNS = [
    "buffer", "node", "table", "queue", "stream", "packet", "frame",
    "chunk", "cache", "token", "cursor", "widget", "matrix", "vector",
    "string", "record", "field", "index", "slot", "page", "block",
    "handle", "socket", "channel", "event", "signal", "timer", "clock",
    "arena", "pool", "heap", "stack", "list", "tree", "graph", "edge",
    "route", "batch", "shard", "bucket",
]

VERBS = [
    "init", "reset", "update", "flush", "parse", "scan", "emit", "load",
    "store", "copy", "move", "swap", "merge", "split", "grow", "shrink",
    "insert", "remove", "find", "count", "hash", "pack", "unpack",
    "encode", "decode", "open", "close", "read", "write", "seek",
    "bind", "attach", "detach", "resize", "clamp", "rotate",
]

ADJS = [
    "tiny", "fast", "slow", "raw", "packed", "sparse", "dense", "local",
    "shared", "dirty", "clean", "hot", "cold", "lazy", "eager", "safe",
    "weak", "strong", "inner", "outer", "upper", "lower", "prime",
    "spare", "stale", "fresh", "wide", "narrow", "deep", "flat",
    "round", "plain",
]

_INT_TYPES = ["int", "long", "unsigned", "unsigned long"]


# ---------------------------------------------------------------------------
# small helpers (all take rng explicitly)
# ---------------------------------------------------------------------------

def _fresh(rng, used, *pools):
    """Join one word from each pool into a fresh snake_case identifier."""
    name = "_".join(rng.choice(p) for p in pools)
    for _ in range(24):
        if name not in used:
            used.add(name)
            return name
        name = "_".join(rng.choice(p) for p in pools)
    name = name + str(rng.randint(2, 97))
    used.add(name)
    return name


def _int_lit(rng):
    r = rng.random()
    if r < 0.35:
        return str(rng.randint(1, 9999))
    if r < 0.60:
        return "0x%x" % rng.randint(1, 0xffff)
    return str(rng.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]))


def _flt_lit(rng):
    return "%d.%d" % (rng.randint(0, 99), rng.randint(0, 99))


def _comment(rng, line=None):
    """A short varied comment; line=True -> //, False -> block, None -> random."""
    if line is None:
        line = rng.random() < 0.5
    text = rng.choice([
        "{v} the {a} {n} before reuse",
        "{n} {n2}: single {v} pass over the {a} region",
        "TODO: {v} the {n} once the {n2} is {a}",
        "NOTE: keep the {n} {a}; see {v}_{n2}()",
        "{a} {n} helpers ({v} / {v2} paths)",
        "internal {n} layout, callers must not {v} it directly",
        "{v} at most {k} {n}s per {n2}",
    ]).format(
        v=rng.choice(VERBS), v2=rng.choice(VERBS),
        n=rng.choice(NOUNS), n2=rng.choice(NOUNS),
        a=rng.choice(ADJS), k=rng.randint(2, 64),
    )
    return "// " + text if line else "/* " + text + " */"


def _helper_fn(rng, used, line_comments=None, use_printf=False):
    """A small stand-alone function used to pad files to a target size."""
    name = _fresh(rng, used, VERBS, ADJS, NOUNS)
    out = []
    if rng.random() < 0.7:
        out.append(_comment(rng, line_comments))
    shape = rng.randint(0, 2 + (1 if use_printf else 0))
    if shape == 0:
        ret = rng.choice(_INT_TYPES)
        out += [
            "static %s %s(%s seed, int rounds) {" % (ret, name, ret),
            "    %s acc = seed ^ %s;" % (ret, _int_lit(rng)),
            "    for (int i = 0; i < rounds; ++i) {",
            "        acc = acc * %s + (%s)i;" % (_int_lit(rng), ret),
            "        acc ^= acc >> %d;" % rng.randint(1, 13),
            "    }",
            "    return acc;",
            "}",
        ]
    elif shape == 1:
        out += [
            "static int %s(const unsigned char *bytes, int n) {" % name,
            "    int total = 0;",
            "    for (int i = 0; i + 1 < n; i += 2) {",
            "        total += (bytes[i] << %d) ^ bytes[i + 1];" % rng.randint(1, 7),
            "        total &= %s;" % _int_lit(rng),
            "    }",
            "    return total;",
            "}",
        ]
    elif shape == 2:
        out += [
            "static double %s(double base, int steps) {" % name,
            "    double acc = base;",
            "    while (steps-- > 0) {",
            "        acc = acc * %s + %s;" % (_flt_lit(rng), _flt_lit(rng)),
            "    }",
            "    return acc < %s ? acc : %s;" % (_flt_lit(rng), _flt_lit(rng)),
            "}",
        ]
    else:
        out += [
            "static void %s(int level, long value) {" % name,
            "    if (level > %s) {" % rng.randint(0, 9),
            "        printf(\"%s=%ld\\n\", \"" + rng.choice(NOUNS) + "\", value);",
            "    }",
            "}",
        ]
    return "\n".join(out) + "\n"


def _proto(rng, used, tag):
    """A declaration line for header-style samples."""
    name = _fresh(rng, used, VERBS, NOUNS)
    kind = rng.randint(0, 3)
    lines = []
    if rng.random() < 0.4:
        lines.append(_comment(rng, False))
    if kind == 0:
        lines.append("int %s_%s(struct %s *self, %s value);"
                     % (tag, name, tag, rng.choice(_INT_TYPES)))
    elif kind == 1:
        lines.append("void %s_%s(struct %s *self, size_t limit);"
                     % (tag, name, tag))
    elif kind == 2:
        lines.append("typedef int (*%s_fn)(void *ctx, long %s);"
                     % (name, rng.choice(NOUNS)))
    else:
        lines.append("extern const unsigned %s_%s[%s];"
                     % (tag, _fresh(rng, used, ADJS, NOUNS), _int_lit(rng)))
    return "\n".join(lines) + "\n"


def _assemble(rng, head, extra_maker, tail=()):
    """Join head blocks, pad with extra blocks toward a random target size,
    append tail, and clamp the result to ~4 KB of ASCII."""
    target = rng.choice([
        rng.randint(260, 700),
        rng.randint(700, 1800),
        rng.randint(1800, 3600),
    ])
    blocks = list(head)
    tail = list(tail)

    def render():
        return "\n".join(blocks + tail) + "\n"

    text = render()
    guard = 0
    while len(text) < target and guard < 64:
        blocks.append(extra_maker(rng))
        text = render()
        guard += 1
    while len(text) > 4000 and len(blocks) > len(head):
        blocks.pop()
        text = render()
    if len(text) > 4090:
        text = text[:4090]
        cut = text.rfind("\n}")
        if cut > 0:
            text = text[:cut + 2] + "\n"
    return text.encode("ascii")


# ---------------------------------------------------------------------------
# hard-C sub-styles: real C wearing C++ clothing
# ---------------------------------------------------------------------------

def _c_style_vtable(rng, used):
    """OO-looking C: typedef'd struct + function-pointer vtable + new_/delete_."""
    n = _fresh(rng, used, NOUNS)
    ops = n + "_ops"
    m1, m2 = rng.sample(VERBS, 2)
    f1 = _fresh(rng, used, ADJS, NOUNS)
    f2 = _fresh(rng, used, ADJS, NOUNS)
    head = [
        _comment(rng, False),
        "#include <stdlib.h>",
    ]
    if rng.random() < 0.6:
        head.append("#include <string.h>")
    head += [
        "",
        "typedef struct %s %s;" % (n, n),
        "",
        "typedef struct %s {" % ops,
        "    void (*destroy)(%s *self);" % n,
        "    int (*%s)(%s *self, int depth);" % (m1, n),
        "    long (*%s)(const %s *self);" % (m2, n),
        "} %s;" % ops,
        "",
        "struct %s {" % n,
        "    const %s *ops;" % ops,
        "    int refcount;",
        "    long %s;" % f1,
        "    double %s;" % f2,
        "};",
        "",
        "%s *new_%s(const %s *ops) {" % (n, n, ops),
        "    %s *self = (%s *)malloc(sizeof(*self));" % (n, n),
        "    if (!self) return NULL;",
        "    self->ops = ops;",
        "    self->refcount = 1;",
        "    self->%s = %s;" % (f1, _int_lit(rng)),
        "    self->%s = %s;" % (f2, _flt_lit(rng)),
        "    return self;",
        "}",
        "",
        "void delete_%s(%s *self) {" % (n, n),
        "    if (self && --self->refcount == 0) {",
        "        self->ops->destroy(self);",
        "        free(self);",
        "    }",
        "}",
        "",
        "int %s_%s(%s *self, int depth) {" % (n, m1, n),
        "    return self->ops->%s(self, depth);" % m1,
        "}",
        "",
    ]
    return head, lambda r: _helper_fn(r, used, line_comments=False)


def _c_style_container(rng, used):
    """Class-like C container: struct string/vector with method functions."""
    n = rng.choice(["string", "vector", "list", "map", "deque"])
    used.add(n)
    grow = rng.choice([2, 2, 4])
    verb = rng.choice(["append", "push", "insert", "emit"])
    head = [
        _comment(rng, True),
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        "struct %s {" % n,
        "    char *data;",
        "    size_t size;",
        "    size_t capacity;",
        "};",
        "",
        "struct %s *new_%s(size_t hint) {" % (n, n),
        "    struct %s *self = malloc(sizeof *self);" % n,
        "    if (!self) return NULL;",
        "    self->size = 0;",
        "    self->capacity = hint ? hint : %s;" % _int_lit(rng),
        "    self->data = malloc(self->capacity);",
        "    return self;",
        "}",
        "",
        "void delete_%s(struct %s *self) {" % (n, n),
        "    if (self) {",
        "        free(self->data);",
        "        free(self);",
        "    }",
        "}",
        "",
        "static int %s_reserve(struct %s *self, size_t want) {" % (n, n),
        "    if (want <= self->capacity) return 0;",
        "    while (self->capacity < want) self->capacity *= %d;" % grow,
        "    self->data = realloc(self->data, self->capacity);",
        "    return self->data ? 0 : -1;",
        "}",
        "",
        "int %s_%s(struct %s *self, const char *bytes, size_t n) {" % (n, verb, n),
        "    if (%s_reserve(self, self->size + n) != 0) return -1;" % n,
        "    memcpy(self->data + self->size, bytes, n);",
        "    self->size += n;",
        "    return 0;",
        "}",
        "",
    ]
    return head, lambda r: _helper_fn(r, used)


def _c_style_stdbool(rng, used):
    """Modern C: stdbool.h, // comments, true/false, fixed-width ints."""
    state = _fresh(rng, used, ADJS, NOUNS) + "_state"
    pred = _fresh(rng, used, NOUNS)
    verb = _fresh(rng, used, VERBS, NOUNS)
    mask = "0x%xu" % rng.randint(1, 0xffff)
    head = [
        _comment(rng, True),
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "    uint32_t flags;",
        "    int32_t depth;",
        "    bool active;",
        "} %s;" % state,
        "",
        "static bool %s_is_%s(const %s *s) {" % (pred, rng.choice(ADJS), state),
        "    return s->active && (s->flags & %s) != 0;" % mask,
        "}",
        "",
        "bool %s(%s *s, int32_t step) {" % (verb, state),
        "    if (!s) return false;",
        "    // clamp the walk so the %s stays %s" % (rng.choice(NOUNS), rng.choice(ADJS)),
        "    for (int32_t i = 0; i < step; ++i) {",
        "        s->depth += (i & 1) ? %d : -%d;" % (rng.randint(1, 9), rng.randint(1, 9)),
        "    }",
        "    s->active = s->depth < %s;" % _int_lit(rng),
        "    return true;",
        "}",
        "",
    ]
    return head, lambda r: _helper_fn(r, used, line_comments=True)


def _c_style_macro_template(rng, used):
    """Poor-man's templates: a type-parameterized container built with ##."""
    mac = rng.choice(["VEC", "ARR", "SPAN", "SLAB", "RING"])
    used.add(mac)
    push = rng.choice(["push", "append", "emit"])
    cap0 = rng.choice([4, 8, 16])
    insts = rng.sample(["int", "long", "double", "float", "short"], rng.randint(2, 3))
    head = [
        "/* generic %s via macros; a poor man's template */" % rng.choice(NOUNS),
        "#include <stdlib.h>",
        "",
        "#define %s(T) \\" % mac,
        "    struct %s_##T { T *data; size_t len; size_t cap; }; \\" % mac.lower(),
        "    static void %s_##T##_init(struct %s_##T *v) { \\" % (mac.lower(), mac.lower()),
        "        v->data = 0; v->len = 0; v->cap = 0; \\",
        "    } \\",
        "    static int %s_##T##_%s(struct %s_##T *v, T item) { \\" % (mac.lower(), push, mac.lower()),
        "        if (v->len == v->cap) { \\",
        "            size_t next = v->cap ? v->cap * 2 : %d; \\" % cap0,
        "            T *p = (T *)realloc(v->data, next * sizeof(T)); \\",
        "            if (!p) return -1; \\",
        "            v->data = p; v->cap = next; \\",
        "        } \\",
        "        v->data[v->len++] = item; \\",
        "        return 0; \\",
        "    }",
        "",
    ] + ["%s(%s)" % (mac, t) for t in insts] + [""]
    return head, lambda r: _helper_fn(r, used)


def _c_style_opaque_header(rng, used):
    """Opaque-handle C header: guard, typedefs, callback prototypes."""
    n = _fresh(rng, used, NOUNS)
    guard = ("%s_%s_H" % (rng.choice(ADJS), n)).upper()
    cb = _fresh(rng, used, VERBS)
    head = [
        "#ifndef %s" % guard,
        "#define %s" % guard,
        "",
        _comment(rng, False),
        "#include <stddef.h>",
        "",
        "typedef struct %s %s;  /* opaque */" % (n, n),
        "typedef void (*%s_%s_fn)(%s *self, void *ctx);" % (n, cb, n),
        "",
        "%s *%s_create(size_t capacity);" % (n, n),
        "void %s_destroy(%s *self);" % (n, n),
        "int %s_%s(%s *self, %s_%s_fn cb, void *ctx);" % (n, cb, n, n, cb),
        "size_t %s_%s(const %s *self);" % (n, rng.choice(["size", "count", "depth"]), n),
        "",
    ]
    tail = ["#endif /* %s */" % guard]
    return head, (lambda r: _proto(r, used, n)), tail


def synth_hard_c(rng):
    """Return bytes of a file that is clearly C but near the C++ boundary."""
    used = set()
    style = rng.choice([
        _c_style_vtable,
        _c_style_container,
        _c_style_stdbool,
        _c_style_macro_template,
        _c_style_opaque_header,
    ])
    parts = style(rng, used)
    if len(parts) == 3:
        head, extra, tail = parts
    else:
        head, extra = parts
        tail = ()
    return _assemble(rng, head, extra, tail)


# ---------------------------------------------------------------------------
# hard-C++ sub-styles: real C++ wearing C clothing, but with 2-4 clear
# C++ markers near the top of every file (C++ includes, class with
# ctor/dtor, std:: usage, new/delete, templates, a lone std::cout).
# ---------------------------------------------------------------------------

def _cpp_helper_fn(rng, used, vector_ok=False):
    """C-flavored padding function for C++ files; some shapes lean on
    new[]/delete[], std::printf, or std::vector to keep C++ evidence up."""
    name = _fresh(rng, used, VERBS, ADJS, NOUNS)
    out = []
    if rng.random() < 0.7:
        out.append(_comment(rng))
    shapes = [0, 1, 2]
    if vector_ok:
        shapes.append(3)
    shape = rng.choice(shapes)
    if shape == 0:
        ret = rng.choice(_INT_TYPES)
        out += [
            "static %s %s(%s seed, int rounds) {" % (ret, name, ret),
            "    %s acc = seed ^ %s;" % (ret, _int_lit(rng)),
            "    for (int i = 0; i < rounds; ++i) {",
            "        acc = acc * %s + (%s)i;" % (_int_lit(rng), ret),
            "        acc ^= acc >> %d;" % rng.randint(1, 13),
            "    }",
            "    return acc;",
            "}",
        ]
    elif shape == 1:
        out += [
            "static long %s(int n) {" % name,
            "    long *tmp = new long[n > 0 ? n : 1];",
            "    long total = 0;",
            "    for (int i = 0; i < n; ++i) {",
            "        tmp[i] = (long)i * %d;" % rng.randint(2, 97),
            "        total += tmp[i];",
            "    }",
            "    delete[] tmp;",
            "    return total;",
            "}",
        ]
    elif shape == 2:
        out += [
            "static void %s(int level, long value) {" % name,
            "    if (level > %d) {" % rng.randint(0, 9),
            "        std::printf(\"%s=%ld\\n\", \"" + rng.choice(NOUNS) + "\", value);",
            "    }",
            "}",
        ]
    else:
        out += [
            "static long %s(const std::vector<long> &items) {" % name,
            "    long total = 0;",
            "    for (std::size_t i = 0; i < items.size(); ++i) {",
            "        total += items[i] ^ %s;" % _int_lit(rng),
            "    }",
            "    return total;",
            "}",
        ]
    return "\n".join(out) + "\n"


def _cpp_style_class_ns(rng, used):
    """A real class (ctor/dtor, public:/private:) inside a namespace, with
    C-style factory functions using new/delete and std::printf."""
    ns = rng.choice(ADJS)
    n = _fresh(rng, used, NOUNS)
    cls = n.capitalize()
    f1 = _fresh(rng, used, ADJS, NOUNS)
    verb = rng.choice(VERBS)
    head = [
        _comment(rng, False),
        "#include <cstdio>",
        "#include <cstdlib>",
        "",
        "namespace %s {" % ns,
        "",
        "class %s {" % cls,
        "public:",
        "    explicit %s(int %s) : %s_(%s), scale_(%s) {}" % (cls, f1, f1, f1, _flt_lit(rng)),
        "    ~%s() { std::printf(\"%s freed: %%d\\n\", %s_); }" % (cls, n, f1),
        "",
        "    int %s(int step) {" % verb,
        "        %s_ += step;" % f1,
        "        return %s_;" % f1,
        "    }",
        "",
        "private:",
        "    int %s_;" % f1,
        "    double scale_;",
        "};",
        "",
        "static %s *make_%s(int seed) {" % (cls, n),
        "    return new %s(seed %% %d);" % (cls, rng.randint(3, 977)),
        "}",
        "",
        "static void drop_%s(%s *p) {" % (n, cls),
        "    delete p;",
        "}",
        "",
    ]
    tail = ["} // namespace %s" % ns]
    return head, (lambda r: _cpp_helper_fn(r, used)), tail


def _cpp_style_extern_c(rng, used):
    """extern \"C\" API over C-style structs, implemented with std::memcpy
    and new[]/delete[] buffers."""
    n = _fresh(rng, used, NOUNS)
    verb = rng.choice(VERBS)
    f1 = _fresh(rng, used, ADJS, NOUNS)
    head = [
        _comment(rng, True),
        "#include <cstring>",
        "#include <cstdio>",
        "",
        "extern \"C\" {",
        "",
        "struct %s_header {" % n,
        "    unsigned magic;",
        "    unsigned %s;" % f1,
        "    unsigned char payload[%s];" % _int_lit(rng),
        "};",
        "",
        "int %s_%s(const unsigned char *bytes, std::size_t len, struct %s_header *out) {" % (verb, n, n),
        "    if (len < sizeof(*out)) return -1;",
        "    std::memcpy(out, bytes, sizeof(*out));",
        "    if (out->magic != %su) return -2;" % _int_lit(rng),
        "    out->%s &= %s;" % (f1, _int_lit(rng)),
        "    return 0;",
        "}",
        "",
        "} /* extern \"C\" */",
        "",
        "static unsigned char *clone_%s(const unsigned char *src, std::size_t n) {" % n,
        "    unsigned char *copy = new unsigned char[n];",
        "    std::memcpy(copy, src, n);",
        "    return copy;",
        "}",
        "",
        "static void free_%s(unsigned char *p) {" % n,
        "    delete[] p;",
        "}",
        "",
    ]
    return head, (lambda r: _cpp_helper_fn(r, used)), ()


def _cpp_style_template(rng, used):
    """Mostly-C procedural code fronted by small template <typename T>
    helpers and std::size_t."""
    n = _fresh(rng, used, NOUNS)
    v1, v2 = rng.sample(VERBS, 2)
    lim = _int_lit(rng)
    head = [
        _comment(rng, rng.random() < 0.5),
        "#include <cstdio>",
        "#include <cstddef>",
        "",
        "template <typename T>",
        "static T %s_%s(T value, T lo, T hi) {" % (v1, n),
        "    if (value < lo) return lo;",
        "    return value > hi ? hi : value;",
        "}",
        "",
        "template <typename T>",
        "static T %s_%s(const T *items, std::size_t n) {" % (v2, n),
        "    T total = T();",
        "    for (std::size_t i = 0; i < n; ++i) total += items[i];",
        "    return total;",
        "}",
        "",
        "struct %s_span {" % n,
        "    const long *data;",
        "    std::size_t len;",
        "};",
        "",
        "static long fold_%s(const %s_span *s) {" % (n, n),
        "    long total = %s_%s(s->data, s->len);" % (v2, n),
        "    return %s_%s<long>(total, -%s, %s);" % (v1, n, lim, lim),
        "}",
        "",
    ]
    return head, (lambda r: _cpp_helper_fn(r, used)), ()


def _cpp_style_vector_lite(rng, used):
    """Procedural C-style functions that pass std::vector / std::string
    around but otherwise read like C (printf-style output, raw loops)."""
    n = _fresh(rng, used, NOUNS)
    verb = rng.choice(VERBS)
    head = [
        _comment(rng, True),
        "#include <cstdio>",
        "#include <vector>",
        "#include <string>",
        "",
        "struct %s_stats {" % n,
        "    long total;",
        "    int count;",
        "};",
        "",
        "static %s_stats %s_%s(const std::vector<long> &samples) {" % (n, verb, n),
        "    %s_stats st;" % n,
        "    st.total = 0;",
        "    st.count = (int)samples.size();",
        "    for (int i = 0; i < st.count; ++i) st.total += samples[i];",
        "    return st;",
        "}",
        "",
        "static void dump_%s(const std::string &tag, const %s_stats *st) {" % (n, n),
        "    std::printf(\"%s: total=%ld count=%d\\n\", tag.c_str(), st->total, st->count);",
        "}",
        "",
    ]
    return head, (lambda r: _cpp_helper_fn(r, used, vector_ok=True)), ()


def _cpp_style_mixed_alloc(rng, used):
    """malloc-style pool code that also uses new[]/delete[] and finishes
    with a single std::cout line."""
    n = _fresh(rng, used, NOUNS)
    pool = n + "_" + rng.choice(["pool", "arena", "bank"])
    push = rng.choice(["push", "add", "put"])
    head = [
        _comment(rng, rng.random() < 0.5),
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <iostream>",
        "",
        "struct %s {" % pool,
        "    long *slots;",
        "    int used;",
        "    int cap;",
        "};",
        "",
        "static %s *make_%s(int cap) {" % (pool, n),
        "    %s *p = (%s *)malloc(sizeof(%s));" % (pool, pool, pool),
        "    if (!p) return NULL;",
        "    p->slots = new long[cap > 0 ? cap : %d];" % rng.choice([8, 16, 32]),
        "    p->used = 0;",
        "    p->cap = cap;",
        "    return p;",
        "}",
        "",
        "static void drop_%s(%s *p) {" % (n, pool),
        "    if (!p) return;",
        "    delete[] p->slots;",
        "    free(p);",
        "}",
        "",
        "static int %s_%s(%s *p, long v) {" % (n, push, pool),
        "    if (p->used == p->cap) return -1;",
        "    p->slots[p->used++] = v;",
        "    return p->used;",
        "}",
        "",
        "static void report_%s(const %s *p) {" % (n, pool),
        "    std::cout << \"%s used=\" << p->used << \"\\n\";" % n,
        "}",
        "",
    ]
    return head, (lambda r: _cpp_helper_fn(r, used)), ()


def synth_hard_cpp(rng):
    """Return bytes of a file that is clearly C++ but near the C boundary."""
    used = set()
    style = rng.choice([
        _cpp_style_class_ns,
        _cpp_style_extern_c,
        _cpp_style_template,
        _cpp_style_vector_lite,
        _cpp_style_mixed_alloc,
    ])
    head, extra, tail = style(rng, used)
    return _assemble(rng, head, extra, tail)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_rng = random.Random(0)
    for fn in (synth_hard_c, synth_hard_cpp):
        for _ in range(2):
            print(fn(demo_rng).decode("ascii"))
            print("=" * 40)

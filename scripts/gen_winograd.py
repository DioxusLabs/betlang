#!/usr/bin/env python3
"""Generate src/model/wino.rs: Winograd (Cook-Toom) transform kernels.

Requires wincnn.py (https://github.com/andravin/wincnn) on sys.path for the
exact rational Cook-Toom matrices, and sympy.

Each conv stage uses F(4, k) minimal filtering. Interpolation points come in
+-a pairs, so the generated input/output transforms factor into shared
even/odd halves: each point pair costs one even and one odd combination plus
an add/sub, roughly halving the transform arithmetic. Weight (G) matrices
are emitted as constants and applied once at model load.

fp32 error against direct convolution with ternary-scale weights and
activation magnitudes like this model's is ~1e-6 (F(4,3)/F(4,5)) to ~1e-5
(F(4,7)), far below the ~1e-4 error of the runtime's Pade GELU
approximation. Verified by `forward_matches_naive_oracle_on_fuzzed_unit_lengths` in src/model/tests.rs.

Usage:
    python3 scripts/gen_winograd.py > src/model/wino.rs
"""

import sys
from fractions import Fraction

import sympy
from wincnn import cookToomFilter

CONFIGS = [
    ("w0", 4, 7, [0, 1, -1, 2, -2, sympy.Rational(1, 2), -sympy.Rational(1, 2), 3, -3]),
    ("w1", 4, 5, [0, 1, -1, 2, -2, sympy.Rational(1, 2), -sympy.Rational(1, 2)]),
    ("w2", 4, 3, [0, 1, -1, 2, -2]),
]


def flit(v):
    f = float(v)
    if f == int(f) and abs(f) < 1e9:
        return f"{int(f)}.0"
    return repr(f)


def emit_linear(dst, terms):
    ones = [s for s, c in terms if c == 1]
    negs = [s for s, c in terms if c == -1]
    scaled = [(s, c) for s, c in terms if c not in (1, -1)]
    expr = None
    if ones:
        expr = ones[0]
        for s in ones[1:]:
            expr = f"({expr} + {s})"
    for s in negs:
        expr = f"({expr} - {s})" if expr is not None else f"(f32x4::splat(simd, 0.0) - {s})"
    stmts = []
    if expr is None:
        if not scaled:
            return [f"let {dst} = f32x4::splat(simd, 0.0);"]
        s, c = scaled[0]
        stmts.append(f"let mut {dst} = {s} * f32x4::splat(simd, {flit(c)});")
        scaled = scaled[1:]
    else:
        stmts.append(f"let mut {dst} = {expr};")
    for s, c in scaled:
        stmts.append(f"{dst} = {s}.mul_add(f32x4::splat(simd, {flit(c)}), {dst});")
    if len(stmts) == 1 and stmts[0].startswith(f"let mut {dst}"):
        stmts[0] = stmts[0].replace("let mut ", "let ", 1)
    return stmts


def strip_parens(stmts):
    out = []
    for s in stmts:
        prefix, _, rest = s.partition(" = ")
        rest = rest.rstrip(";")
        if rest.startswith("(") and rest.endswith(")"):
            depth = 0
            balanced = True
            for i, ch in enumerate(rest):
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0 and i < len(rest) - 1:
                    balanced = False
                    break
            if balanced:
                rest = rest[1:-1]
        out.append(f"{prefix} = {rest};")
    return out


def gen_input_fn(name, BT, n, pair_rows, single_rows):
    lines = [
        "#[inline(always)]",
        f"pub(crate) fn {name}<S: Simd>(simd: S, d: &[f32x4<S>; {n}]) -> [f32x4<S>; {n}] {{",
    ]
    out = [None] * n
    tmp = 0
    for rp, rm in pair_rows:
        e_terms, o_terms = [], []
        for i in range(n):
            cp, cm = Fraction(str(BT[rp, i])), Fraction(str(BT[rm, i]))
            e, o = (cp + cm) / 2, (cp - cm) / 2
            if e != 0:
                e_terms.append((f"d[{i}]", e))
            if o != 0:
                o_terms.append((f"d[{i}]", o))
        ev, ov = f"e{tmp}", f"o{tmp}"
        tmp += 1
        lines += ["    " + s for s in strip_parens(emit_linear(ev, e_terms))]
        lines += ["    " + s for s in strip_parens(emit_linear(ov, o_terms))]
        lines.append(f"    let r{rp} = {ev} + {ov};")
        lines.append(f"    let r{rm} = {ev} - {ov};")
        out[rp], out[rm] = f"r{rp}", f"r{rm}"
    for r in single_rows:
        terms = [(f"d[{i}]", Fraction(str(BT[r, i]))) for i in range(n) if BT[r, i] != 0]
        lines += ["    " + s for s in strip_parens(emit_linear(f"r{r}", terms))]
        out[r] = f"r{r}"
    lines.append(f"    [{', '.join(out)}]")
    lines.append("}")
    return "\n".join(lines)


def gen_output_fn(name, AT, n, m, pair_cols, single_cols):
    lines = [
        "#[inline(always)]",
        f"pub(crate) fn {name}<S: Simd>(simd: S, m: &[f32x4<S>; {n}]) -> [f32x4<S>; {m}] {{",
    ]
    sym = {}
    tmp = 0
    for jp, jm in pair_cols:
        lines.append(f"    let s{tmp} = m[{jp}] + m[{jm}];")
        lines.append(f"    let t{tmp} = m[{jp}] - m[{jm}];")
        sym[jp] = (f"s{tmp}", f"t{tmp}")
        tmp += 1
    outs = []
    for p in range(m):
        terms = [(f"m[{j}]", Fraction(str(AT[p, j]))) for j in single_cols if AT[p, j] != 0]
        for jp, jm in pair_cols:
            cp, cm = Fraction(str(AT[p, jp])), Fraction(str(AT[p, jm]))
            s_c, t_c = (cp + cm) / 2, (cp - cm) / 2
            sname, tname = sym[jp]
            if s_c != 0:
                terms.append((sname, s_c))
            if t_c != 0:
                terms.append((tname, t_c))
        lines += ["    " + s for s in strip_parens(emit_linear(f"y{p}", terms))]
        outs.append(f"y{p}")
    lines.append(f"    [{', '.join(outs)}]")
    lines.append("}")
    return "\n".join(lines)


def main():
    parts = []
    parts.append("//! Winograd transform kernels, generated from exact Cook-Toom matrices")
    parts.append("//! (`scripts/gen_winograd.py`). Do not edit by hand.")
    parts.append("//!")
    parts.append("//! Interpolation points come in +-a pairs, so the transforms factor into")
    parts.append("//! shared even/odd halves: each point pair costs one even and one odd")
    parts.append("//! combination plus an add/sub, roughly halving the arithmetic.")
    parts.append("//!")
    parts.append("//! Points per stage:")
    for name, m, r, pts in CONFIGS:
        parts.append(
            f"//! - `{name}`: F({m},{r}) over {{{', '.join(str(p) for p in pts)}}} (plus infinity)"
        )
    parts.append("")
    parts.append("#![allow(clippy::excessive_precision)]")
    parts.append("")
    parts.append("use fearless_simd::{Simd, SimdBase, SimdFloat, f32x4};")
    parts.append("")

    for name, m, r, pts in CONFIGS:
        AT, G, BT, _ = cookToomFilter(tuple(pts), m, r)
        n = m + r - 1
        finite = len(pts)
        pairs = [(i, i + 1) for i in range(1, finite, 2)]
        singles = [0, n - 1]
        parts.append(f"pub(crate) const {name.upper()}_POINTS: usize = {n};")
        parts.append(f"pub(crate) const {name.upper()}_G: [[f32; {r}]; {n}] = [")
        for i in range(n):
            parts.append(f"    [{', '.join(flit(G[i, j]) for j in range(r))}],")
        parts.append("];")
        parts.append("")
        parts.append(gen_input_fn(f"input_{name}", BT, n, pairs, singles))
        parts.append("")
        parts.append(gen_output_fn(f"output_{name}", AT, n, m, pairs, singles))
        parts.append("")

    sys.stdout.write("\n".join(parts))


if __name__ == "__main__":
    main()

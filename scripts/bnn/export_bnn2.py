#!/usr/bin/env python3
"""Export trained BNN2/BNN2Seg models (optionally an ensemble) to a packed
bitwise artifact ("BBN2" v2) + bit-exact integer reference.

Inference semantics (all integer/bitwise):
  input: 2 bitplanes (shannon bits, codeword boundaries), each BITS long
  conv layer with m binary bases: for basis j, z'_j = 2*match_j - n_valid
    (XNOR+popcount); z_q = sum_j aq[j] * z'_j (i64, aq = round(alpha*FIX))
  fire iff (sign==1) ? z_q >= thr : z_q <= thr    (thr i32, fixed-point)
  OR-pool window 4; segmented popcount head; i16 head dot + f32 scale/bias.

Layout (little-endian):
  u32 magic="BBN2", u32 version=2, u32 BITS, u32 CLASSES, u32 n_models
  256 x u8 shannon code lengths
  per model:
    u32 C, STEM_K, STEM_STRIDE, SEGS
    3 x layer:  u32 m, u32 cin, u32 k, then per basis:
                  cout x ceil(cin*k/32) u32 weight bits (bit = tap*cin + ci),
                  cout x i32 aq
                then cout x u8 sign, cout x i32 thr
    head: CLASSES x (SEGS*C) i16, CLASSES x f32 scale, CLASSES x f32 bias
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import BITS, CLASSES, OUT
from train_bnn2 import BNN2, BinConvM, C, STEM_K, STEM_STRIDE, build_plane_cache, unpack_batch
from finetune_seg import BNN2Seg, SEGS as SEGS8

MAGIC = 0x324E4242  # "BBN2"
FIX = 65536  # alpha fixed-point scale (2^16)


def conv_export(conv: BinConvM):
    """Returns (wsigns [m, cout, k*cin] bit=tap*cin+ci, aq [m, cout] i32)."""
    m, cout, cin, k = conv.latent.shape
    std = conv.latent.detach().std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    wb = (conv.latent.detach() - conv.shift.detach().view(m, 1, 1, 1) * std) >= 0
    signs = wb.numpy().astype(np.uint8)                    # [m, cout, cin, k]
    signs = np.transpose(signs, (0, 1, 3, 2)).reshape(m, cout, k * cin)
    aq = np.round(conv.alpha.detach().numpy() * FIX).astype(np.int64)
    return signs, aq, cin, k


def fold_thresholds_q(bn, rsign_b):
    """Thresholds on z_q = FIX * (float pre-BN activation)."""
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mu = bn.running_mean.detach().numpy()
    sigma = np.sqrt(bn.running_var.detach().numpy() + bn.eps)
    b = rsign_b.detach().numpy().reshape(-1)
    lhs = gamma / sigma
    rhs = b - beta + gamma * mu / sigma
    t = rhs / np.where(np.abs(lhs) < 1e-12, 1e-12, lhs) * FIX
    sign = np.where(lhs >= 0, 1, 0).astype(np.uint8)
    thr = np.where(sign == 1, np.ceil(t), np.floor(t))
    thr = np.clip(thr, -(2 ** 30), 2 ** 30)
    return sign, thr.astype(np.int64)


def head_export(hb, fc):
    gamma = hb.weight.detach().numpy()
    beta = hb.bias.detach().numpy()
    mu = hb.running_mean.detach().numpy()
    sigma = np.sqrt(hb.running_var.detach().numpy() + hb.eps)
    W = fc.weight.detach().numpy()
    b = fc.bias.detach().numpy()
    A = W * (gamma / sigma)[None, :]
    Bc = b + W @ (beta - mu * gamma / sigma)
    scale = np.abs(A).max(axis=1) / 32767.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(A / scale[:, None]), -32767, 32767).astype(np.int16)
    return q, scale.astype(np.float32), Bc.astype(np.float32)


def pack_bits_u32(signs: np.ndarray) -> np.ndarray:
    rows, nbits = signs.shape
    words = (nbits + 31) // 32
    out = np.zeros((rows, words), dtype=np.uint32)
    for j in range(nbits):
        out[:, j // 32] |= (signs[:, j].astype(np.uint32) << np.uint32(j % 32))
    return out


class IntLayer:
    def __init__(self, conv: BinConvM, bn, rsign):
        self.w, self.aq, self.cin, self.k = conv_export(conv)
        self.sg, self.th = fold_thresholds_q(bn, rsign.b)
        self.stride = conv.stride
        self.pad = conv.padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: [cin, L] {0,1} -> [cout, Lout] {0,1}; integer exact."""
        cin, L = x.shape
        m, cout, _ = self.w.shape
        k, stride, pad = self.k, self.stride, self.pad
        Lout = (L + 2 * pad - k) // stride + 1
        zq = np.zeros((cout, Lout), dtype=np.int64)
        cols = np.zeros((k, cin, Lout), dtype=np.int64)
        valid = np.zeros((k, Lout), dtype=np.int64)
        for tap in range(k):
            src = np.arange(Lout) * stride - pad + tap
            ok = (src >= 0) & (src < L)
            valid[tap] = ok
            cols[tap][:, ok] = x[:, src[ok]]
        nv = (valid * cin).sum(0)
        for j in range(m):
            wj = self.w[j].reshape(cout, k, cin)
            match = np.zeros((cout, Lout), dtype=np.int64)
            for tap in range(k):
                wb = wj[:, tap, :].astype(np.int64)
                xb = cols[tap]
                eq = wb @ xb + (1 - wb) @ (1 - xb)
                eq -= np.outer((1 - wb).sum(1), 1 - valid[tap])
                match += eq
            zq += self.aq[j][:, None] * (2 * match - nv[None, :])
        return np.where(self.sg[:, None] == 1, zq >= self.th[:, None],
                        zq <= self.th[:, None]).astype(np.uint8)


class IntModel:
    def __init__(self, model, weight: float = 1.0):
        self.segs = model.fc.in_features // C
        self.l0 = IntLayer(model.stem, model.bn0, model.s0)
        self.l1 = IntLayer(model.conv1, model.bn1, model.s1)
        self.l2 = IntLayer(model.conv2, model.bn2, model.s2)
        self.hq, self.hs, self.hb_ = head_export(model.hb, model.fc)
        self.hs = (self.hs * weight).astype(np.float32)
        self.hb_ = (self.hb_ * weight).astype(np.float32)

    def logits(self, planes01: np.ndarray) -> np.ndarray:
        h = self.l0.forward(planes01)
        L = (h.shape[1] // 4) * 4
        h = h[:, :L].reshape(h.shape[0], -1, 4).max(2)
        h = self.l1.forward(h)
        h = h.reshape(h.shape[0], -1, 4).max(2)
        h = self.l2.forward(h)
        segs = self.segs
        seg = h.shape[1] // segs
        counts = np.concatenate([h[:, i * seg:(i + 1) * seg].sum(1)
                                 for i in range(segs)])
        acc = self.hq.astype(np.int64) @ counts.astype(np.int64)
        return acc * self.hs + self.hb_

    def write(self, buf: bytearray):
        buf += struct.pack("<4I", C, STEM_K, STEM_STRIDE, self.segs)
        for layer in (self.l0, self.l1, self.l2):
            m, cout, nbits = layer.w.shape
            buf += struct.pack("<3I", m, layer.cin, layer.k)
            for j in range(m):
                buf += pack_bits_u32(layer.w[j]).tobytes()
                buf += layer.aq[j].astype(np.int32).tobytes()
            buf += layer.sg.tobytes()
            buf += layer.th.astype(np.int32).tobytes()
        buf += self.hq.tobytes()
        buf += self.hs.tobytes()
        buf += self.hb_.tobytes()


ENSEMBLE = [("best2seg.pt", 0.7), ("best3.pt", 0.7),
            ("best2seg_synth.ep0.pt", 1.0), ("best2seg_synth3.ep0.pt", 1.6),
            ("best3_synth2.ep0.pt", 1.3)]


def load_models() -> list:
    """Returns [(model, weight)] with weights normalized so that the mean of
    the weighted logits equals the weighted-softmax-temperature convention
    logits_total / sum(weights) used during weight search."""
    wsum = sum(w for _, w in ENSEMBLE)
    k = len(ENSEMBLE)
    models = []
    for ck, w in ENSEMBLE:
        cls = BNN2Seg if "2seg" in ck else BNN2
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.eval()
        models.append((m, w * k / wsum))
    return models


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    pairs = load_models()
    models = [m for m, _ in pairs]
    ints = [IntModel(m, w) for m, w in pairs]

    va_bits, va_labels, _ = build_plane_cache("valid", codes)
    n_par = 128
    idx = np.arange(0, len(va_labels), max(1, len(va_labels) // n_par))[:n_par]
    x = unpack_batch(va_bits, idx)
    agree = [0] * len(models)
    with torch.no_grad():
        refs = [m(x).numpy() for m in models]
    for row, i in enumerate(idx):
        planes = np.unpackbits(np.asarray(va_bits[i]), axis=1)
        for mi, im in enumerate(ints):
            got = im.logits(planes)
            if got.argmax() == refs[mi][row].argmax():
                agree[mi] += 1
    print("int-vs-torch argmax agreement:", [f"{a}/{len(idx)}" for a in agree])

    buf = bytearray()
    buf += struct.pack("<5I", MAGIC, 2, BITS, CLASSES, len(ints))
    buf += lengths256.astype(np.uint8).tobytes()
    for im in ints:
        im.write(buf)
    path = OUT / "source-bnn2.bin"
    path.write_bytes(bytes(buf))
    print(f"wrote {path} ({len(buf)} bytes)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the 3D models for the energy-extraction design.

Two assemblies are built procedurally, in metres with +Y up:

  models/eel-habitat-harvester.*      Part A - passive habitat harvester
  models/artificial-electric-organ.*  Part B - eel-inspired hydrogel power cell

Each one is written as
  .glb        glTF 2.0 binary (materials, hierarchy, part descriptions in extras)
  .obj + .mtl Wavefront (one object per part)
  .stl        binary STL, millimetres, +Z up, opaque parts only

    python tools/build_models.py

Only numpy is required. Output is deterministic.
"""
import json
import math
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models")


# --------------------------------------------------------------------------
# Mesh primitives
# --------------------------------------------------------------------------

class Mesh:
    def __init__(self, pos, nrm, idx):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        self.nrm = np.asarray(nrm, dtype=np.float64).reshape(-1, 3)
        self.idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)

    def xform(self, M):
        M = np.asarray(M, dtype=np.float64)
        pos = self.pos @ M[:3, :3].T + M[:3, 3]
        nrm = self.nrm @ np.linalg.inv(M[:3, :3])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        idx = self.idx[:, ::-1] if np.linalg.det(M[:3, :3]) < 0 else self.idx
        return Mesh(pos, nrm, idx)


def merge(meshes):
    pos, nrm, idx, off = [], [], [], 0
    for m in meshes:
        pos.append(m.pos)
        nrm.append(m.nrm)
        idx.append(m.idx + off)
        off += len(m.pos)
    return Mesh(np.vstack(pos), np.vstack(nrm), np.vstack(idx))


def T(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def R(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    if axis == "x":
        M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    elif axis == "y":
        M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    else:
        M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def S(x, y, z):
    return np.diag([x, y, z, 1.0])


def chain(*Ms):
    out = np.eye(4)
    for M in Ms:
        out = out @ M
    return out


def vertex_normals(pos, idx):
    fn = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    vn = np.zeros_like(pos)
    for c in range(3):
        np.add.at(vn, idx[:, c], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(ln > 1e-15, vn / np.maximum(ln, 1e-15), np.array([0.0, 1.0, 0.0]))
    return vn


def flat(mesh):
    """Unweld so every triangle is flat shaded (low-poly look)."""
    tri = mesh.pos[mesh.idx]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-12          # sphere poles leave zero-area triangles
    tri, fn = tri[keep], fn[keep] / area[keep, None]
    p = tri.reshape(-1, 3)
    return Mesh(p, np.repeat(fn, 3, axis=0), np.arange(len(p)).reshape(-1, 3))


# (axis, sign, u, v) with u x v = outward normal
_BOX_FACES = [(0, 1, 1, 2), (0, -1, 2, 1), (1, 1, 2, 0), (1, -1, 0, 2), (2, 1, 0, 1), (2, -1, 1, 0)]


def box(size, center=(0.0, 0.0, 0.0)):
    h = np.asarray(size, dtype=float) / 2
    c = np.asarray(center, dtype=float)
    E = np.eye(3)
    pos, nrm, idx = [], [], []
    for a, s, u, v in _BOX_FACES:
        n = E[a] * s
        U, V = E[u] * h[u], E[v] * h[v]
        base = c + n * h[a]
        k = len(pos)
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pos.append(base + du * U + dv * V)
            nrm.append(n)
        idx += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return Mesh(pos, nrm, idx)


def _ring(r, y, ang):
    return np.stack([r * np.cos(ang), np.full_like(ang, y), r * np.sin(ang)], axis=1)


def cylinder(r, h, seg=32, caps=True, r_top=None):
    """Cylinder (or cone frustum) along +Y, centred on the origin."""
    r_top = r if r_top is None else r_top
    ang = np.linspace(0, 2 * np.pi, seg + 1)
    bot, top = _ring(r, -h / 2, ang), _ring(r_top, h / 2, ang)
    n = np.stack([np.cos(ang), np.full_like(ang, (r - r_top) / h), np.sin(ang)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    pos = np.empty((2 * (seg + 1), 3))
    pos[0::2], pos[1::2] = bot, top
    nrm = np.repeat(n, 2, axis=0)
    idx = []
    for i in range(seg):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        idx += [(a, b, d), (a, d, c)]
    parts = [Mesh(pos, nrm, idx)]
    if caps:
        parts.append(_disc(r, -h / 2, ang[:-1], -1))
        if r_top > 0:
            parts.append(_disc(r_top, h / 2, ang[:-1], 1))
    return merge(parts)


def _disc(r, y, ang, sign):
    ring = _ring(r, y, ang)
    pos = np.vstack([[0.0, y, 0.0], ring])
    nrm = np.tile([0.0, float(sign), 0.0], (len(pos), 1))
    n = len(ang)
    if sign > 0:
        idx = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
    else:
        idx = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return Mesh(pos, nrm, idx)


def sphere(r, seg=24, rings=16, hemi=False):
    phis = np.linspace(0, np.pi / 2 if hemi else np.pi, rings + 1)
    th = np.linspace(0, 2 * np.pi, seg + 1)
    P, Th = np.meshgrid(phis, th, indexing="ij")
    n = np.stack([np.sin(P) * np.cos(Th), np.cos(P), np.sin(P) * np.sin(Th)], axis=-1).reshape(-1, 3)
    idx = []
    w = seg + 1
    for i in range(rings):
        for j in range(seg):
            a, b = i * w + j, (i + 1) * w + j
            idx += [(a, b + 1, b), (a, a + 1, b + 1)]
    m = Mesh(n * r, n, idx)
    if hemi:
        m = merge([m, _disc(r, 0.0, th[:-1], -1)])
    return m


def pipe(r_in, r_out, h, seg=40):
    """Open tube along +Y with annular end faces."""
    outer = cylinder(r_out, h, seg, caps=False)
    inner = cylinder(r_in, h, seg, caps=False)
    inner = Mesh(inner.pos, -inner.nrm, inner.idx[:, ::-1])
    ang = np.linspace(0, 2 * np.pi, seg + 1)[:-1]
    parts = [outer, inner]
    for y, sign in ((h / 2, 1), (-h / 2, -1)):
        o, i_ = _ring(r_out, y, ang), _ring(r_in, y, ang)
        pos = np.vstack([o, i_])
        nrm = np.tile([0.0, float(sign), 0.0], (len(pos), 1))
        idx = []
        for k in range(seg):
            k1 = (k + 1) % seg
            tri = [(k, seg + k, seg + k1), (k, seg + k1, k1)]
            idx += tri if sign > 0 else [t[::-1] for t in tri]
        parts.append(Mesh(pos, nrm, idx))
    return merge(parts)


def cyl_between(a, b, r, seg=16, caps=True):
    """Cylinder whose axis runs from point a to point b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return cylinder(r, np.linalg.norm(d), seg, caps).xform(T(*((a + b) / 2)) @ align_y(d))


def align_y(d):
    """Rotation taking +Y onto direction d (Rodrigues)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(y, d), float(np.dot(y, d))
    M = np.eye(4)
    if np.linalg.norm(v) < 1e-9:
        if c < 0:
            M[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return M
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    M[:3, :3] = np.eye(3) + vx + vx @ vx / (1 + c)
    return M


def arrow(a, b, r_shaft, r_head, head_len, seg=24):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    neck = b - d * head_len
    head = cylinder(r_head, head_len, seg, caps=True, r_top=0.0)
    head = head.xform(T(*((neck + b) / 2)) @ align_y(d))
    return merge([cyl_between(a, neck, r_shaft, seg), head])


# --------------------------------------------------------------------------
# Paths and sweeps
# --------------------------------------------------------------------------

def resample(P, step):
    P = np.asarray(P, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9])
    P = P[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def rounded_path(pts, radius=0.06, step=0.012):
    """Polyline with every corner filleted by a quadratic Bezier."""
    P = np.asarray(pts, float)
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        d1, d2 = a - b, c - b
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        r = min(radius, 0.45 * l1, 0.45 * l2)
        s, e = b + d1 / l1 * r, b + d2 / l2 * r
        for t in np.linspace(0, 1, 9):
            out.append((1 - t) ** 2 * s + 2 * (1 - t) * t * b + t * t * e)
    out.append(P[-1])
    return resample(np.array(out), step)


def frames(P, world_up=False):
    """Tangent / normal / binormal per sample (projected transport)."""
    Tn = np.gradient(P, axis=0)
    Tn /= np.maximum(np.linalg.norm(Tn, axis=1, keepdims=True), 1e-15)
    up = np.array([0.0, 1.0, 0.0])
    N = np.zeros_like(P)
    for i in range(len(P)):
        ref = up if (world_up or i == 0) else N[i - 1]
        n = ref - Tn[i] * np.dot(ref, Tn[i])
        if np.linalg.norm(n) < 1e-6:
            alt = np.array([1.0, 0.0, 0.0]) if abs(Tn[i][0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            n = alt - Tn[i] * np.dot(alt, Tn[i])
        N[i] = n / np.linalg.norm(n)
    B = np.cross(Tn, N)
    return Tn, N, B


def _fan(center, ring, normal):
    pos = np.vstack([center, ring])
    nrm = np.tile(normal, (len(pos), 1))
    n = len(ring)
    idx = np.array([(0, 1 + i, 1 + (i + 1) % n) for i in range(n)])
    f = np.cross(pos[idx[0, 1]] - pos[0], pos[idx[0, 2]] - pos[0])
    if np.dot(f, normal) < 0:
        idx = idx[:, ::-1]
    return Mesh(pos, nrm, idx)


def sweep(P, profiles, world_up=False, caps=True):
    """Sweep closed 2D profiles (n, k, 2) given as (side, up) offsets along P."""
    P = np.asarray(P, float)
    Tn, N, B = frames(P, world_up)
    n, k = profiles.shape[:2]
    pos = P[:, None, :] + profiles[..., :1] * B[:, None, :] + profiles[..., 1:2] * N[:, None, :]
    pos = pos.reshape(-1, 3)
    idx = []
    for i in range(n - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            idx += [(a, b, b + k), (a, b + k, a + k)]
    idx = np.array(idx)
    mid = n // 2
    faces = idx[mid * k * 2:(mid + 1) * k * 2]
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    centroid = pos[faces].mean(axis=1)
    if np.sum(fn * (centroid - P[mid])) < 0:
        idx = idx[:, ::-1]
    parts = [Mesh(pos, vertex_normals(pos, idx), idx)]
    if caps:
        parts.append(_fan(P[0], pos[:k], -Tn[0]))
        parts.append(_fan(P[-1], pos[-k:], Tn[-1]))
    return merge(parts)


def tube(path, r, seg=10):
    P = np.asarray(path, float)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    prof = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    return sweep(P, np.repeat(prof[None], len(P), axis=0))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lattice(w, h, t, frame=0.02, bar=0.008, pitch=0.045):
    """Flat guard grid in the local XY plane, thickness along Z."""
    parts = [box((w, frame, t), (0, s * (h - frame) / 2, 0)) for s in (-1, 1)]
    parts += [box((frame, h - 2 * frame, t), (s * (w - frame) / 2, 0, 0)) for s in (-1, 1)]
    iw, ih = w - 2 * frame, h - 2 * frame
    for x in np.linspace(-iw / 2, iw / 2, int(iw // pitch) + 1)[1:-1]:
        parts.append(box((bar, ih, t * 0.8), (x, 0, 0)))
    for y in np.linspace(-ih / 2, ih / 2, int(ih // pitch) + 1)[1:-1]:
        parts.append(box((iw, bar, t * 0.8), (0, y, 0)))
    return merge(parts)


# --------------------------------------------------------------------------
# Scene graph and materials
# --------------------------------------------------------------------------

class Node:
    def __init__(self, name, mesh=None, mat=None, desc=None, children=None, stl=True):
        self.name, self.mesh, self.mat, self.desc = name, mesh, mat, desc
        self.children = children or []
        self.stl = stl

    def leaves(self, prefix=""):
        path = f"{prefix}{self.name}"
        if self.mesh is not None:
            yield path, self
        for c in self.children:
            yield from c.leaves(path + "/")


def mat(color, alpha=1.0, metal=0.0, rough=0.5, emit=None, double=False):
    return {"color": color, "alpha": alpha, "metal": metal, "rough": rough,
            "emit": emit or (0.0, 0.0, 0.0), "double": double}


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


MATERIALS = {
    # Part A
    "cabinet_wood": mat((0.30, 0.21, 0.15), rough=0.6),
    "cabinet_dark": mat((0.06, 0.06, 0.06), rough=0.5),
    "steel_brushed": mat((0.78, 0.78, 0.80), metal=1.0, rough=0.3),
    "glass": mat((0.80, 0.93, 0.92), alpha=0.10, rough=0.04),
    "aluminium_black": mat((0.05, 0.05, 0.06), metal=0.6, rough=0.4),
    "water": mat((0.36, 0.55, 0.46), alpha=0.12, rough=0.03),
    "sand": mat((0.72, 0.62, 0.47), rough=0.95),
    "rock": mat((0.42, 0.40, 0.37), rough=0.9),
    "pvc_hide": mat((0.33, 0.29, 0.24), rough=0.7),
    "lid_mesh": mat((0.08, 0.08, 0.08), rough=0.6),
    "eel_skin": mat((0.27, 0.25, 0.19), rough=0.38),
    "eel_fin": mat((0.21, 0.18, 0.15), rough=0.5, double=True),
    "eel_eye": mat((0.02, 0.02, 0.02), rough=0.05),
    "field_glow": mat((1.0, 0.72, 0.12), rough=0.4, emit=(1.0, 0.62, 0.05)),
    "graphite": mat((0.20, 0.20, 0.22), metal=0.3, rough=0.55),
    "guard_acrylic": mat((0.90, 0.93, 0.95), rough=0.3),
    "standoff_pvc": mat((0.92, 0.92, 0.90), rough=0.5),
    "cable": mat((0.05, 0.05, 0.05), rough=0.55),
    "steel_cabinet": mat((0.30, 0.33, 0.36), metal=0.4, rough=0.45),
    "enclosure_abs": mat((0.83, 0.84, 0.85), rough=0.5),
    "polycarbonate": mat((0.88, 0.93, 0.97), alpha=0.22, rough=0.05, double=True),
    "pcb_green": mat((0.08, 0.42, 0.20), rough=0.4),
    "pcb_blue": mat((0.10, 0.22, 0.55), rough=0.4),
    "component_black": mat((0.04, 0.04, 0.04), rough=0.35),
    "terminal_green": mat((0.15, 0.58, 0.32), rough=0.45),
    "screw_metal": mat((0.82, 0.82, 0.84), metal=1.0, rough=0.25),
    "film_cap_yellow": mat((0.97, 0.78, 0.15), rough=0.4),
    "mov_blue": mat((0.15, 0.38, 0.78), rough=0.4),
    "supercap_sleeve": mat((0.12, 0.18, 0.45), rough=0.3),
    "sensor_white": mat((0.94, 0.94, 0.92), rough=0.45),
    "probe_steel": mat((0.75, 0.76, 0.78), metal=1.0, rough=0.3),
    "led_green": mat((0.2, 1.0, 0.4), emit=(0.1, 0.9, 0.3)),
    "beacon_amber": mat((1.0, 0.62, 0.12), alpha=0.85, rough=0.15, emit=(1.0, 0.45, 0.0)),
    "display_black": mat((0.04, 0.04, 0.045), rough=0.4),
    "epaper": mat((0.88, 0.88, 0.84), rough=0.85),
    "epaper_ink": mat((0.06, 0.06, 0.06), rough=0.85),
    # Part B
    "base_plate": mat((0.86, 0.86, 0.87), rough=0.6),
    "gel_high_salt": mat((0.92, 0.42, 0.32), rough=0.35),
    "gel_low_salt": mat((0.62, 0.80, 0.95), rough=0.35),
    "membrane_cation": mat((0.50, 0.38, 0.82), rough=0.45),
    "membrane_anion": mat((0.96, 0.70, 0.20), rough=0.45),
    "silicone_clear": mat((0.92, 0.96, 1.0), alpha=0.18, rough=0.1, double=True),
    "silicone_white": mat((0.93, 0.93, 0.91), rough=0.6),
    "silver": mat((0.85, 0.85, 0.87), metal=1.0, rough=0.25),
    "titanium": mat((0.62, 0.62, 0.64), metal=1.0, rough=0.35),
    "epoxy_header": mat((0.86, 0.92, 0.92), alpha=0.55, rough=0.1),
    "lead_positive": mat((0.85, 0.15, 0.12), rough=0.5),
    "lead_negative": mat((0.10, 0.12, 0.15), rough=0.5),
    "ion_cation": mat((0.98, 0.48, 0.10), rough=0.4, emit=(0.35, 0.12, 0.0)),
    "ion_anion": mat((0.22, 0.72, 0.38), rough=0.4, emit=(0.05, 0.25, 0.08)),
    "current_arrow": mat((1.0, 0.86, 0.25), rough=0.4, emit=(0.6, 0.45, 0.05)),
    "pin_steel": mat((0.7, 0.7, 0.72), metal=1.0, rough=0.35),
}


# --------------------------------------------------------------------------
# Part A - passive eel habitat harvester
# --------------------------------------------------------------------------

G = 0.019                      # glass thickness
IN_X, IN_Z = 1.5 - G, 0.6 - G  # inner half-length / half-depth of the tank
SUB_TOP = 0.80 + G + 0.04      # top of the substrate
WATER_TOP = 1.68               # leaves a 12 cm air gap: eels breathe air
LID_Y = 1.806
EY = 1.27                      # electrode centre height
E0 = np.array([1.95, 1.05, -0.25])  # harvester enclosure origin (bottom centre)


def eel_spine(n=240):
    s = np.linspace(0, 1, 600)
    x = 0.48 - 1.50 * s
    z = 0.07 + (0.03 + 0.13 * s) * np.sin(2 * np.pi * 1.55 * s + 0.4)
    y = 1.34 - 0.09 * s + 0.025 * np.sin(np.pi * s)
    return resample(np.stack([x, y, z], axis=1), 1.0 / n * 1.6)


def build_eel():
    P = eel_spine()
    L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    t = np.linspace(0, 1, len(P))
    R0 = 0.05
    snout = np.sqrt(np.clip(1 - (1 - np.clip(t / 0.07, 0, 1)) ** 2, 0, 1))
    taper = np.where(t < 0.3, 1.0, np.clip((1 - t) / 0.7, 0, 1) ** 0.85)
    r = np.maximum(R0 * snout * taper, 0.0015)
    sx = np.interp(t, [0, 0.08, 0.25, 1.0], [1.22, 1.15, 1.0, 0.6])   # half-width factor
    sy = np.interp(t, [0, 0.08, 0.25, 1.0], [0.80, 0.86, 1.0, 1.3])   # half-height factor
    k = 28
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    prof = np.stack([np.cos(ang)[None] * (r * sx)[:, None], np.sin(ang)[None] * (r * sy)[:, None]], axis=-1)
    body = sweep(P, prof, world_up=True)

    # Anal fin: the long ventral ribbon knifefish swim with.
    sel = (t >= 0.12)
    tf, Pf = t[sel], P[sel]
    hf = (r * sy)[sel]
    f = 0.03 * smoothstep(0.12, 0.26, tf) * (0.35 + 0.65 * (1 - smoothstep(0.85, 1.0, tf)))
    th = 0.0018
    top = -hf * 0.7
    bot = -hf - f
    fin_prof = np.stack([
        np.stack([-th * np.ones_like(tf), top], axis=1),
        np.stack([th * np.ones_like(tf), top], axis=1),
        np.stack([th * np.ones_like(tf), bot], axis=1),
        np.stack([-th * np.ones_like(tf), bot], axis=1),
    ], axis=1)
    fin = sweep(Pf, fin_prof, world_up=True)

    Tn, N, B = frames(P, world_up=True)
    i = int(0.045 * (len(P) - 1))
    eyes = []
    for side in (-1, 1):
        c = P[i] + B[i] * side * r[i] * sx[i] * 0.86 + N[i] * r[i] * sy[i] * 0.42
        eyes.append(sphere(0.0065, 16, 10).xform(T(*c)))
    return P, r, L, body, fin, merge(eyes)


def field_lines(P, r):
    """Illustrative head(+) to tail(-) dipole field lines that stay in the water."""
    n = len(P)
    H = P[int(0.05 * (n - 1))]
    Tl = P[int(0.97 * (n - 1))]
    lo = np.array([-IN_X + 0.08, SUB_TOP + 0.03, -IN_Z + 0.10])
    hi = np.array([IN_X - 0.08, WATER_TOP - 0.03, IN_Z - 0.03])
    body_P = P[int(0.15 * n):int(0.85 * n)]
    body_r = r[int(0.15 * n):int(0.85 * n)]

    def E(p):
        a, b = p - H, p - Tl
        v = a / np.linalg.norm(a) ** 3 - b / np.linalg.norm(b) ** 3
        return v / np.linalg.norm(v)

    golden = math.pi * (3 - math.sqrt(5))
    lines = []
    for j in range(160):
        yv = 1 - 2 * (j + 0.5) / 160
        rad = math.sqrt(1 - yv * yv)
        d = np.array([math.cos(golden * j) * rad, yv, math.sin(golden * j) * rad])
        p = H + 0.07 * d
        pts = [p.copy()]
        ok = False
        for _ in range(1500):
            k1 = E(p)
            p = p + 0.006 * E(p + 0.003 * k1)
            pts.append(p.copy())
            if np.any(p < lo) or np.any(p > hi):
                break
            if np.linalg.norm(p - Tl) < 0.07:
                ok = True
                break
        if not ok:
            continue
        pts = np.array(pts)
        dist = np.linalg.norm(pts[:, None, :] - body_P[None], axis=2)
        if np.any(dist - body_r[None] < 0.02):
            continue
        reach = float(np.max(np.linalg.norm(np.cross(pts - H, (Tl - H) / np.linalg.norm(Tl - H)), axis=1)))
        lines.append((reach, pts))
    lines.sort(key=lambda x: x[0])
    if len(lines) > 12:
        pick = np.linspace(0, len(lines) - 1, 12).round().astype(int)
        lines = [lines[i] for i in sorted(set(pick))]
    meshes = [tube(resample(pts, 0.015), 0.0024, 6) for _, pts in lines]
    return merge(meshes), len(meshes)


def build_electrodes():
    """Six graphite plates, each behind a perforated acrylic guard."""
    specs = [("E1", (-IN_X + 0.016, EY, 0.0), 90)]
    specs += [(f"E{i + 2}", (xp, EY, -IN_Z + 0.016), 0) for i, xp in enumerate((-1.05, -0.35, 0.35, 1.05))]
    specs += [("E6", (IN_X - 0.016, EY, 0.0), -90)]
    PW, PH, PT = 0.30, 0.45, 0.008
    plates, guards, stand = [], [], []
    tops = []
    for name, c, rot in specs:
        M = T(*c) @ R("y", rot)
        plates.append(box((PW, PH, PT)).xform(M))
        for sx_ in (-1, 1):
            for sy_ in (-1, 1):
                stand.append(cylinder(0.006, 0.012, 12).xform(M @ T(sx_ * 0.12, sy_ * 0.19, -0.010) @ R("x", 90)))
                stand.append(cylinder(0.007, 0.072, 12).xform(M @ T(sx_ * 0.18, sy_ * 0.255, 0.020) @ R("x", 90)))
        guards.append(lattice(0.38, 0.53, 0.006).xform(M @ T(0, 0, 0.059)))
        tops.append((name, (M @ np.array([0, PH / 2, 0, 1]))[:3], rot))
    return merge(plates), merge(guards), merge(stand), tops


def electrode_cables(tops):
    """Route each plate lead over the rim and into a gland on the harvester."""
    paths = []
    gx = E0[0] - 0.234
    for i, (name, top, rot) in enumerate(tops):
        zg = E0[2] - 0.10 + 0.04 * i
        lane_z = -0.64 - 0.011 * i
        lane_y = 1.872 + 0.004 * (i % 2)
        tail = [(1.40, lane_y, lane_z), (1.60, 1.80, -0.52 + 0.025 * i), (1.66, 1.35, zg), (gx - 0.02, 1.12, zg), (gx, 1.12, zg)]
        if rot == 0:                       # back wall plates
            x = top[0]
            head = [top + (0, 0.002, 0), (x, 1.70, -IN_Z + 0.011), (x, 1.85, -0.60), (x + 0.06, lane_y, lane_z)]
            pts = head + tail
        elif rot == 90:                    # left end wall
            head = [top + (0, 0.002, 0), (-IN_X + 0.007, 1.70, 0.0), (-1.50, 1.86, 0.0), (-1.55, lane_y, -0.05),
                    (-1.55, lane_y, lane_z), (-1.45, lane_y, lane_z)]
            pts = head + tail
        else:                              # right end wall
            pts = [top + (0, 0.002, 0), (IN_X - 0.007, 1.70, 0.0), (1.50, 1.86, 0.0), (1.56, 1.86, -0.05),
                   (1.64, 1.50, zg), (1.66, 1.35, zg), (gx - 0.02, 1.12, zg), (gx, 1.12, zg)]
        paths.append(rounded_path([np.asarray(p, float) for p in pts], radius=0.07))
    return merge([tube(p, 0.0045, 10) for p in paths])


def build_harvester():
    """Enclosure on a pedestal; parts in enclosure-local coordinates."""
    M = T(*E0)
    parts = {}
    parts["Pedestal"] = (box((0.45, 1.05, 0.45), (E0[0], 0.525, E0[2])), "steel_cabinet")
    tray = [box((0.42, 0.006, 0.32), (0, 0.003, 0))]
    tray += [box((0.42, 0.12, 0.006), (0, 0.06, s * 0.157)) for s in (-1, 1)]
    tray += [box((0.006, 0.12, 0.308), (s * 0.207, 0.06, 0)) for s in (-1, 1)]
    parts["Enclosure"] = (merge(tray).xform(M), "enclosure_abs")
    parts["Clear lid"] = (box((0.42, 0.006, 0.32), (0, 0.123, 0)).xform(M), "polycarbonate")
    standoffs = [cylinder(0.004, 0.014, 12).xform(T(sx_ * 0.17, 0.013, sz_ * 0.12)) for sx_ in (-1, 1) for sz_ in (-1, 1)]
    parts["PCB standoffs"] = (merge(standoffs).xform(M), "screw_metal")
    pcb_y = 0.0208
    parts["Main board"] = (box((0.38, 0.0016, 0.28), (0, pcb_y - 0.0008, 0)).xform(M), "pcb_green")

    term_in = box((0.02, 0.016, 0.24), (-0.17, pcb_y + 0.008, 0))
    term_out = box((0.02, 0.016, 0.09), (0.17, pcb_y + 0.008, 0.08))
    parts["Terminal blocks"] = (merge([term_in, term_out]).xform(M), "terminal_green")
    screws = [cylinder(0.004, 0.003, 12).xform(T(-0.17, pcb_y + 0.0175, -0.10 + 0.04 * i)) for i in range(6)]
    screws += [cylinder(0.004, 0.003, 12).xform(T(0.17, pcb_y + 0.0175, 0.05 + 0.03 * i)) for i in range(3)]
    parts["Terminal screws"] = (merge(screws).xform(M), "screw_metal")

    # 6-input polyphase bridge: two diodes per plate, one to each rail.
    diodes = []
    for i in range(6):
        z = -0.10 + 0.04 * i
        for x in (-0.125, -0.095):
            diodes.append(cylinder(0.0035, 0.014, 12).xform(T(x, pcb_y + 0.0035, z) @ R("z", 90)))
    parts["Polyphase bridge (12 diodes)"] = (merge(diodes).xform(M), "component_black")
    movs = [cylinder(0.012, 0.005, 24).xform(T(-0.055, pcb_y + 0.013, z) @ R("x", 90)) for z in (-0.05, 0.0)]
    parts["Surge clamp (MOV + TVS)"] = (merge(movs).xform(M), "mov_blue")
    parts["Pulse catch capacitor"] = (box((0.058, 0.036, 0.022), (0.0, pcb_y + 0.018, -0.07)).xform(M), "film_cap_yellow")
    parts["Buck converter board"] = (box((0.05, 0.002, 0.035), (0.055, pcb_y + 0.006, 0.05)).xform(M), "pcb_blue")
    buck = [cylinder(0.009, 0.008, 24).xform(T(0.045, pcb_y + 0.011, 0.05)),
            box((0.008, 0.002, 0.008), (0.068, pcb_y + 0.008, 0.045)),
            box((0.016, 0.002, 0.016), (0.135, pcb_y + 0.001, 0.06)),
            box((0.012, 0.004, 0.008), (0.10, pcb_y + 0.002, -0.02))]
    parts["Converter, MCU and radio"] = (merge(buck).xform(M), "component_black")
    caps = [cylinder(0.0125, 0.035, 24).xform(T(0.10, pcb_y + 0.0175, z)) for z in (-0.085, -0.055)]
    parts["Supercapacitors (2 x 2 F, series)"] = (merge(caps).xform(M), "supercap_sleeve")
    glands = [cylinder(0.008, 0.024, 16).xform(T(-0.222, 0.07, -0.10 + 0.04 * i) @ R("z", 90)) for i in range(6)]
    glands += [cylinder(0.008, 0.024, 16).xform(T(x, 0.07, 0.172) @ R("x", 90)) for x in (0.06, 0.11, 0.16)]
    parts["Cable glands"] = (merge(glands).xform(M), "component_black")
    return parts


def build_loads():
    lid_top = LID_Y + 0.006
    out = {}
    # Eel health monitor: sensor node on the lid + guarded probe in the water.
    node = [box((0.10, 0.045, 0.05), (1.20, lid_top + 0.0225, -0.575))]
    out["Health monitor node"] = (merge(node), "sensor_white")
    out["Radio antenna"] = (cylinder(0.004, 0.07, 12).xform(T(1.235, lid_top + 0.045 + 0.035, -0.575)), "component_black")
    out["Status LED"] = (sphere(0.004, 12, 8).xform(T(1.17, lid_top + 0.03, -0.549)), "led_green")
    out["Probe guard"] = (pipe(0.022, 0.025, 0.22, 24).xform(T(1.25, 1.40, -0.40)), "guard_acrylic")
    out["Temperature + conductivity probe"] = (merge([
        cylinder(0.011, 0.20, 20).xform(T(1.25, 1.41, -0.40)),
        cylinder(0.006, 0.02, 16).xform(T(1.25, 1.30, -0.40)),
    ]), "probe_steel")
    # Keeper shock alarm: beacon on the lid next to the feeding hatch.
    out["Alarm beacon base"] = (cylinder(0.03, 0.025, 24).xform(T(1.40, lid_top + 0.0125, -0.575)), "component_black")
    out["Alarm beacon lens"] = (merge([
        cylinder(0.026, 0.03, 32, caps=False).xform(T(1.40, lid_top + 0.040, -0.575)),
        sphere(0.026, 32, 12, hemi=True).xform(T(1.40, lid_top + 0.055, -0.575)),
    ]), "beacon_amber")
    # Education display in front of the tank, facing visitors.
    DX, DZ = 1.05, 1.02
    out["Display stand"] = (merge([
        cylinder(0.16, 0.015, 40).xform(T(DX, 0.0075, DZ)),
        cylinder(0.018, 1.02, 20).xform(T(DX, 0.525, DZ)),
        box((0.08, 0.05, 0.05), (DX, 1.06, DZ - 0.01)),
    ]), "display_black")
    P = T(DX, 1.17, DZ + 0.02) @ R("x", -15)
    out["Display frame"] = (box((0.46, 0.32, 0.02)).xform(P), "display_black")
    out["E-paper screen"] = (box((0.42, 0.28, 0.003), (0, 0, 0.0105)).xform(P), "epaper")
    ink = [box((0.36, 0.002, 0.001), (0, 0.045, 0.0125))]
    for i, hgt in enumerate((0.07, 0.065, 0.06, 0.05, 0.042, 0.035, 0.028, 0.022, 0.016, 0.011)):
        ink.append(box((0.003, hgt, 0.001), (-0.15 + 0.012 * i, 0.045 + hgt / 2, 0.0125)))
    for i in range(5):
        ink.append(box((0.003, 0.012, 0.001), (0.03 + 0.03 * i, 0.051, 0.0125)))
    ink.append(box((0.36, 0.002, 0.001), (0, -0.115, 0.0125)))
    for i, hgt in enumerate((0.035, 0.06, 0.028, 0.075, 0.05, 0.066, 0.042)):
        ink.append(box((0.03, hgt, 0.001), (-0.15 + 0.05 * i, -0.114 + hgt / 2, 0.0125)))
    out["E-paper content"] = (merge(ink).xform(P), "epaper_ink")
    return out


def load_cables():
    g = lambda x: E0 + np.array([x, 0.07, 0.184])  # noqa: E731 - output gland tip
    lid_top = LID_Y + 0.006
    beacon = [g(0.06), g(0.06) + (0, 0, 0.03), (1.95, 1.55, -0.03), (1.62, 1.90, -0.30), (1.47, 1.90, -0.52), (1.43, lid_top + 0.013, -0.575)]
    node = [g(0.11), g(0.11) + (0, 0, 0.05), (1.99, 1.58, -0.01), (1.62, 1.93, -0.27), (1.30, 1.90, -0.50), (1.25, lid_top + 0.02, -0.575)]
    probe = [(1.25, 1.51, -0.40), (1.25, 1.75, -0.40), (1.25, 1.84, -0.47), (1.22, lid_top + 0.045, -0.565)]
    disp = [g(0.16), g(0.16) + (0, 0, 0.06), (E0[0] + 0.16, 0.02, -0.01), (2.0, 0.006, 0.40), (1.17, 0.006, 0.97)]
    paths = [rounded_path([np.asarray(p, float) for p in pts], radius=0.08) for pts in (beacon, node, probe, disp)]
    return merge([tube(p, 0.0035, 10) for p in paths])


def scene_habitat():
    stats = {}
    cab = Node("Cabinet", merge([
        box((3.04, 0.72, 1.24), (0, 0.42, 0)),
        box((3.08, 0.02, 1.28), (0, 0.79, 0)),
    ]), "cabinet_wood", "Tank stand, 0.8 m high; holds the filtration and heater (mains kit is on an RCD).")
    plinth = Node("Plinth and door seams", merge(
        [box((2.96, 0.06, 1.16), (0, 0.03, 0))]
        + [box((0.006, 0.62, 0.004), (x, 0.43, 0.6205)) for x in (-0.76, 0.0, 0.76)]
    ), "cabinet_dark")
    handles = Node("Door handles", merge(
        [cylinder(0.008, 0.12, 16).xform(T(x, 0.62, 0.628)) for x in (-0.80, -0.72, -0.04, 0.04, 0.72, 0.80)]
    ), "steel_brushed")
    glass = Node("Glass panels", merge([
        box((3.0, 1.0, G), (0, 1.30, 0.6 - G / 2)),
        box((3.0, 1.0, G), (0, 1.30, -0.6 + G / 2)),
        box((G, 1.0, 1.2 - 2 * G), (-1.5 + G / 2, 1.30, 0)),
        box((G, 1.0, 1.2 - 2 * G), (1.5 - G / 2, 1.30, 0)),
        box((3.0 - 2 * G, G, 1.2 - 2 * G), (0, 0.80 + G / 2, 0)),
    ]), "glass", "3.0 x 1.2 x 1.0 m tank, 19 mm glass, about 2,900 L of water.", stl=False)
    trims = []
    for y in (0.815, 1.785):
        trims += [box((3.03, 0.03, 0.03), (0, y, z)) for z in (-0.6, 0.6)]
        trims += [box((0.03, 0.03, 1.17), (x, y, 0)) for x in (-1.5, 1.5)]
    trims += [box((0.03, 0.94, 0.03), (x, 1.30, z)) for x in (-1.5, 1.5) for z in (-0.6, 0.6)]
    frame = Node("Frame trims", merge(trims), "aluminium_black")
    water = Node("Water", box((2.960, WATER_TOP - SUB_TOP, 1.160), (0, (WATER_TOP + SUB_TOP) / 2, 0)), "water",
                 "Soft, warm water. The surface stops 12 cm below the rim: electric eels gulp air and must reach it.",
                 stl=False)
    sand = Node("Substrate", box((2.960, 0.04, 1.160), (0, SUB_TOP - 0.02, 0)), "sand")
    rocks = []
    for (x, z, rr, seed) in ((0.12, -0.33, 0.12, 3), (0.66, -0.30, 0.09, 7), (-0.30, 0.40, 0.08, 11), (1.05, 0.35, 0.07, 5)):
        m = sphere(1.0, 10, 7)
        rng = np.random.default_rng(seed)
        kv = rng.normal(size=(4, 3)) * 2.0
        ph = rng.uniform(0, 2 * np.pi, 4)
        disp = 1 + 0.13 * np.sum(np.sin(m.pos @ kv.T + ph), axis=1)
        p = m.pos * disp[:, None] * rr
        p[:, 1] *= 0.62
        rocks.append(flat(Mesh(p + (x, SUB_TOP + rr * 0.25, z), m.nrm, m.idx)))
    rock = Node("Rocks", merge(rocks), "rock")
    hide = Node("Hide tube", pipe(0.10, 0.11, 0.75, 48).xform(T(-0.85, SUB_TOP + 0.09, -0.28) @ R("z", 90)), "pvc_hide",
                "A dark tube to retreat into. Eels that can hide discharge less from stress.")
    lid_parts = [box((3.0, 0.012, 0.04), (0, LID_Y, z)) for z in (-0.58, 0.58)]
    lid_parts += [box((0.04, 0.012, 1.12), (x, LID_Y, 0)) for x in (-1.48, 1.48)]
    lid_parts += [box((2.92, 0.004, 0.004), (0, LID_Y, z)) for z in np.arange(-0.50, 0.501, 0.10)]
    lid_parts += [box((0.004, 0.004, 1.12), (x, LID_Y, 0)) for x in np.arange(-1.40, 1.401, 0.10)]
    hy = LID_Y + 0.012
    lid_parts += [box((0.5, 0.012, 0.025), (0.80, hy, z)) for z in (-0.18, 0.22)]
    lid_parts += [box((0.025, 0.012, 0.425), (x, hy, 0.02)) for x in (0.55, 1.05)]
    lid_parts += [box((0.08, 0.012, 0.015), (0.80, hy + 0.012, 0.235))]
    lid = Node("Lid with feeding hatch", merge(lid_parts), "lid_mesh",
               "Escape-proof mesh lid. Keepers open the hatch to feed, next to the shock alarm beacon.")
    habitat = Node("Habitat", children=[cab, plinth, handles, glass, frame, water, sand, rock, hide, lid],
                   desc="Welfare comes first: a very large tank, soft warm water, places to hide.")

    P, r, L, body, fin, eyes = build_eel()
    stats["eel_length_m"] = round(L, 3)
    eel = Node("Electric eel", children=[
        Node("Body", body, "eel_skin", "Electrophorus, about 1.7 m. Its electric organ fills roughly 80% of the body: "
             "about 6,000 electrocytes in series, ~0.15 V each, up to 860 V (E. voltai) for ~2 ms."),
        Node("Anal fin", fin, "eel_fin", "The long ventral fin it swims with; there is no tail fin."),
        Node("Eyes", eyes, "eel_eye"),
    ], desc="Lives normally. The machine never stimulates, baits or restrains it.")

    fl, n_lines = field_lines(P, r)
    stats["field_lines"] = n_lines
    field = Node("Discharge field (illustrative)", children=[
        Node("Field lines", fl, "field_glow", "Head-positive dipole field of one high-voltage pulse. Most of the "
             "energy heats the water near the eel; the wall plates intercept only about 0.1-1% of it.", stl=False),
    ])

    plates, guards, stand, tops = build_electrodes()
    electrodes = Node("Electrode array", children=[
        Node("Graphite plates E1-E6", plates, "graphite", "Six inert graphite plates, 300 x 450 mm. No copper or "
             "zinc: their ions are toxic to fish."),
        Node("Guard screens", guards, "guard_acrylic", "Perforated acrylic, 75 mm from the glass. The eel can never "
             "touch an electrode."),
        Node("Standoffs", stand, "standoff_pvc"),
    ], desc="Six plates feed a polyphase bridge, so whichever two plates see the biggest difference conduct, "
            "in either polarity, wherever the eel points.")
    cables = Node("Cabling", children=[
        Node("Electrode leads", electrode_cables(tops), "cable", "One insulated lead per plate, over the rim to the "
             "harvester's cable glands."),
        Node("Load cables", load_cables(), "cable"),
    ])

    hparts = build_harvester()
    hdesc = {
        "Pedestal": "Steel stand that keeps the electronics out of splash range.",
        "Clear lid": "See-through lid so visitors can watch the electronics.",
        "Polyphase bridge (12 diodes)": "Two diodes per plate, one to each rail. The rails always carry the largest "
                                        "plate-to-plate voltage. No switching logic.",
        "Surge clamp (MOV + TVS)": "Clamps the rails at about 150 V; anything above is shed as heat.",
        "Pulse catch capacitor": "47 uF / 250 V film capacitor that catches each 2 ms spike. A volley's worth is spent "
                                 "at once: it flashes the alarm and wakes the logger.",
        "Converter, MCU and radio": "Buck converter (6-150 V in) that holds its input at half the plate voltage, "
                                    "the matched load, plus a low-power MCU and Bluetooth LE radio.",
        "Supercapacitors (2 x 2 F, series)": "Low-leakage, 1 F at up to 5 V (12.5 J max). Banks energy only on active "
                                              "days, for the display. Output is 3.3 V only, so it can never become a "
                                              "shock device.",
    }
    harvester = Node("Harvester", children=[Node(k, m, mt, hdesc.get(k), stl=(mt != "polycarbonate"))
                                            for k, (m, mt) in hparts.items()],
                     desc="Passive: it only absorbs, never drives current into the water.")

    lparts = build_loads()
    groups = {
        "Eel health monitor": ["Health monitor node", "Radio antenna", "Status LED", "Probe guard",
                               "Temperature + conductivity probe"],
        "Keeper shock alarm": ["Alarm beacon base", "Alarm beacon lens"],
        "Education display": ["Display stand", "Display frame", "E-paper screen", "E-paper content"],
    }
    gdesc = {
        "Eel health monitor": "Logs water temperature, conductivity and every discharge the plates detect; a change "
                              "in discharge pattern is an early sign of stress or illness. Runs on ~25 uW from its "
                              "own 10-year lithium cell, so it never depends on a sick eel discharging.",
        "Keeper shock alarm": "Every high-voltage volley flashes the beacon: tank is live, hands out. Powered by the "
                              "very shock it warns about.",
        "Education display": "E-paper panel: energy harvested today, the last pulse waveform, and how long one eel "
                             "would take to charge a phone (about a thousand years). It refreshes only when the "
                             "eel has made enough surplus energy.",
    }
    loads = Node("Loads (right purposes)", children=[
        Node(gname, children=[Node(p, lparts[p][0], lparts[p][1]) for p in plist], desc=gdesc[gname])
        for gname, plist in groups.items()
    ])
    return [habitat, eel, field, electrodes, cables, harvester, loads], stats


# --------------------------------------------------------------------------
# Part B - artificial electric organ (eel-inspired hydrogel stack)
# --------------------------------------------------------------------------

def scene_organ():
    AX = 0.0535          # stack axis height
    UNITS = 20
    layer = [("gel_high_salt", 0.003), ("membrane_cation", 0.0008), ("gel_low_salt", 0.003), ("membrane_anion", 0.0008)]
    unit = sum(t for _, t in layer)
    x = -UNITS * unit / 2
    discs = {k: [] for k, _ in layer}
    for _ in range(UNITS):
        for k, t in layer:
            discs[k].append(cylinder(0.020, t, 48).xform(T(x + t / 2, AX, 0) @ R("z", 90)))
            x += t
    half = UNITS * unit / 2
    names = {
        "gel_high_salt": ("High-salt gel layers", "Hydrogel with concentrated salt, like the ion-rich side of an "
                                                  "electrocyte."),
        "membrane_cation": ("Cation-selective membranes", "Let positive ions (Na+) through, toward +x."),
        "gel_low_salt": ("Low-salt gel layers", "Dilute hydrogel the ions flow into."),
        "membrane_anion": ("Anion-selective membranes", "Let negative ions (Cl-) through, toward -x. Both ion flows "
                                                        "add up to one current."),
    }
    stack = Node("Gel stack (20 cells)", children=[Node(names[k][0], merge(discs[k]), k, names[k][1]) for k in discs],
                 desc=f"{UNITS} four-layer cells in series. At ~0.18 V per cell that is ~3.6 V open circuit, "
                      "enough for a low-power implant through a boost converter.")
    sleeve = Node("Silicone sleeve", pipe(0.0205, 0.0235, 2 * half + 0.006, 48).xform(T(0, AX, 0) @ R("z", 90)),
                  "silicone_clear", "Soft, flexible, biocompatible casing.", stl=False)
    caps = Node("End caps", merge([cylinder(0.0235, 0.003, 48).xform(T(s * (half + 0.0045), AX, 0) @ R("z", 90))
                                   for s in (-1, 1)]), "silicone_white")
    elec = Node("Silver chloride electrodes", merge([cylinder(0.020, 0.0015, 48).xform(T(s * (half + 0.002), AX, 0) @ R("z", 90))
                                             for s in (-1, 1)]), "silver",
                "Reversible silver / silver-chloride electrodes turn the ion current into electron current.")
    cradle = Node("Cradle", merge([box((0.02, 0.022, 0.05), (s * 0.06, 0.019, 0)) for s in (-1, 1)]), "base_plate")

    IZ = 0.062
    implant = Node("Implant (pacemaker-style)", children=[
        Node("Titanium can", cylinder(0.025, 0.0075, 64).xform(T(0, 0.008 + 0.00375, IZ)), "titanium",
             "The load: an implant powered by the body's own salt gradients, so no battery-replacement surgery."),
        Node("Header", box((0.024, 0.0075, 0.012), (0, 0.008 + 0.00375, IZ - 0.028)), "epoxy_header"),
    ])
    hx = half + 0.006
    pos_lead = rounded_path([np.array(p, float) for p in [(hx, AX, 0), (hx + 0.016, AX, 0), (hx + 0.02, 0.025, 0.036),
                                                          (0.04, 0.0125, 0.045), (0.006, 0.0118, IZ - 0.034)]], 0.012, 0.003)
    neg_lead = rounded_path([np.array(p, float) for p in [(-hx, AX, 0), (-hx - 0.016, AX, 0), (-hx - 0.02, 0.025, 0.036),
                                                          (-0.04, 0.0125, 0.045), (-0.006, 0.0118, IZ - 0.034)]], 0.012, 0.003)
    leads = Node("Leads", children=[
        Node("Positive lead (+x end)", tube(pos_lead, 0.0012, 10), "lead_positive"),
        Node("Negative lead (-x end)", tube(neg_lead, 0.0012, 10), "lead_negative"),
    ])

    # Exploded view of one cell: LS | AEM | HS | CEM | LS
    EZ, EYc, ER = 0.075, 0.042, 0.016
    seq = [("gel_low_salt", 0.010), ("membrane_anion", 0.003), ("gel_high_salt", 0.010),
           ("membrane_cation", 0.003), ("gel_low_salt", 0.010)]
    gap = 0.016
    total = sum(t for _, t in seq) + gap * (len(seq) - 1)
    x = -total / 2
    ex = {}
    centers = []
    for k, t in seq:
        ex.setdefault(k, []).append(cylinder(ER, t, 48).xform(T(x + t / 2, EYc, EZ) @ R("z", 90)))
        centers.append(x + t / 2)
        x += t + gap
    pins = merge([cyl_between((cx, 0.008, EZ), (cx, EYc - ER + 0.001, EZ), 0.0012, 10) for cx in centers])
    ay = EYc + ER + 0.008
    exploded = Node("Exploded cell", children=[Node(names[k][0].replace("layers", "layer").replace("membranes", "membrane"),
                                                    merge(v), k) for k, v in ex.items()]
                    + [Node("Display pins", pins, "pin_steel")],
                    desc="One cell pulled apart: Na+ crosses the cation membrane one way, Cl- crosses the anion "
                         "membrane the other way, and both make current flow toward +x.")
    hs_x = centers[2]
    ions = Node("Ion flow", children=[
        Node("Na+ arrow", arrow((hs_x, ay, EZ), (centers[4] + 0.004, ay, EZ), 0.0016, 0.0042, 0.008), "ion_cation"),
        Node("Cl- arrow", arrow((hs_x, ay, EZ), (centers[0] - 0.004, ay, EZ), 0.0016, 0.0042, 0.008), "ion_anion"),
        Node("Current arrow", arrow((-total / 2, ay + 0.016, EZ), (total / 2, ay + 0.016, EZ), 0.0018, 0.0048, 0.009),
             "current_arrow", "Conventional current runs toward +x, so the +x end is the positive terminal."),
        Node("Na+ ions", merge([sphere(0.0025, 12, 8).xform(T(cx, EYc + dy, EZ + dz)) for cx, dy, dz in
                                ((hs_x - 0.002, 0.004, ER * 0.95), (hs_x + 0.003, -0.005, ER * 0.95),
                                 (centers[4], 0.002, ER * 0.95))]), "ion_cation"),
        Node("Cl- ions", merge([sphere(0.0032, 12, 8).xform(T(cx, EYc + dy, EZ + dz)) for cx, dy, dz in
                                ((hs_x + 0.002, 0.0, ER * 0.95), (centers[0], -0.003, ER * 0.95))]), "ion_anion"),
    ])
    # Lay the two exhibits side by side so the arrows never sit in front of the stack.
    for n in (cradle, stack, sleeve, caps, elec, leads, implant):
        shift(n, T(-0.085, 0, -0.01))
    for n in (exploded, ions):
        shift(n, T(0.10, 0, -0.075))
    base = Node("Base plate", box((0.38, 0.008, 0.20), (-0.005, 0.004, 0.0)), "base_plate")
    return [base, cradle, stack, sleeve, caps, elec, leads, implant, exploded, ions], {"cells": UNITS}


def shift(node, M):
    if node.mesh is not None:
        node.mesh = node.mesh.xform(M)
    for c in node.children:
        shift(c, M)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_glb(path, roots, title):
    used = []
    for r in roots:
        for _, leaf in r.leaves():
            if leaf.mat not in used:
                used.append(leaf.mat)
    mindex = {m: i for i, m in enumerate(used)}
    blob = bytearray()
    views, accessors, meshes, nodes = [], [], [], []

    def view(data, target):
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    def leaf_mesh(node):
        pos = node.mesh.pos.astype(np.float32)
        nrm = node.mesh.nrm.astype(np.float32)
        big = len(pos) > 65535
        idx = node.mesh.idx.astype(np.uint32 if big else np.uint16).ravel()
        a_pos = len(accessors)
        accessors.append({"bufferView": view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                          "type": "VEC3", "min": [float(v) for v in pos.min(axis=0)],
                          "max": [float(v) for v in pos.max(axis=0)]})
        accessors.append({"bufferView": view(nrm.tobytes(), 34962), "componentType": 5126, "count": len(nrm),
                          "type": "VEC3"})
        accessors.append({"bufferView": view(idx.tobytes(), 34963), "componentType": 5125 if big else 5123,
                          "count": len(idx), "type": "SCALAR"})
        meshes.append({"name": node.name, "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_pos + 1}, "indices": a_pos + 2,
            "material": mindex[node.mat]}]})
        return len(meshes) - 1

    def add(node):
        i = len(nodes)
        entry = {"name": node.name}
        nodes.append(entry)
        if node.desc:
            entry["extras"] = {"description": node.desc}
        if node.mesh is not None:
            entry["mesh"] = leaf_mesh(node)
        if node.children:
            entry["children"] = [add(c) for c in node.children]
        return i

    root_ids = [add(r) for r in roots]
    mats = []
    for name in used:
        m = MATERIALS[name]
        rgb = [float(v) for v in srgb_to_linear(m["color"])]
        e = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": rgb + [m["alpha"]],
                                                    "metallicFactor": m["metal"], "roughnessFactor": m["rough"]}}
        if any(m["emit"]):
            e["emissiveFactor"] = [float(v) for v in srgb_to_linear(m["emit"])]
        if m["alpha"] < 1:
            e["alphaMode"] = "BLEND"
        if m["double"]:
            e["doubleSided"] = True
        mats.append(e)
    while len(blob) % 4:
        blob.append(0)
    gltf = {
        "asset": {"version": "2.0", "generator": "energy-extraction tools/build_models.py",
                  "extras": {"title": title, "units": "metres, +Y up"}},
        "scene": 0, "scenes": [{"name": title, "nodes": root_ids}],
        "nodes": nodes, "meshes": meshes, "materials": mats,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        f.write(bytes(blob))


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip("_")


def write_obj(path, roots, title):
    base = os.path.splitext(os.path.basename(path))[0]
    out = [f"# {title}", "# Units: metres, +Y up. Generated by tools/build_models.py", f"mtllib {base}.mtl"]
    used, off = [], 1
    for r in roots:
        for p, leaf in r.leaves():
            if leaf.mat not in used:
                used.append(leaf.mat)
            m = leaf.mesh
            out.append(f"o {_safe(p.replace('/', '.'))}")
            out.append("\n".join("v %.5f %.5f %.5f" % tuple(v) for v in m.pos))
            out.append("\n".join("vn %.4f %.4f %.4f" % tuple(v) for v in m.nrm))
            out.append(f"usemtl {leaf.mat}")
            out.append("\n".join("f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c) for a, b, c in (m.idx + off)))
            off += len(m.pos)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    lines = [f"# Materials for {title}"]
    for name in used:
        m = MATERIALS[name]
        c = m["color"]
        spec = c if m["metal"] > 0.5 else (0.5 * (1 - m["rough"]),) * 3
        lines += [f"newmtl {name}", "Ka 0.000 0.000 0.000", "Kd %.3f %.3f %.3f" % tuple(c),
                  "Ks %.3f %.3f %.3f" % tuple(spec), "Ns %.1f" % (10 + 900 * (1 - m["rough"]) ** 2),
                  "d %.3f" % m["alpha"], "Tr %.3f" % (1 - m["alpha"]),
                  "Ke %.3f %.3f %.3f" % tuple(m["emit"]), "Pr %.3f" % m["rough"], "Pm %.3f" % m["metal"],
                  "illum 2", ""]
    with open(os.path.splitext(path)[0] + ".mtl", "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines))


def write_stl(path, roots, title):
    tris = []
    for r in roots:
        for _, leaf in r.leaves():
            if leaf.stl:
                tris.append(leaf.mesh.pos[leaf.mesh.idx])
    t = np.concatenate(tris)                       # (n, 3, 3) metres, +Y up
    t = np.stack([t[..., 0], -t[..., 2], t[..., 1]], axis=-1) * 1000.0   # mm, +Z up
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, t
    with open(path, "wb") as f:
        f.write(f"{title} | mm, +Z up".encode("ascii")[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        f.write(rec.tobytes())
    return len(t)


def export(stem, roots, title):
    base = os.path.join(OUT, stem)
    write_glb(base + ".glb", roots, title)
    write_obj(base + ".obj", roots, title)
    n_stl = write_stl(base + ".stl", roots, title)
    verts = sum(len(l.mesh.pos) for r in roots for _, l in r.leaves())
    tris = sum(len(l.mesh.idx) for r in roots for _, l in r.leaves())
    parts = sum(1 for r in roots for _ in r.leaves())
    sizes = {ext: os.path.getsize(base + ext) for ext in (".glb", ".obj", ".mtl", ".stl")}
    print(f"{stem}: {parts} parts, {verts:,} vertices, {tris:,} triangles (STL {n_stl:,})")
    print("   " + ", ".join(f"{e} {s / 1e6:.2f} MB" for e, s in sizes.items()))


def main():
    os.makedirs(OUT, exist_ok=True)
    roots, stats = scene_habitat()
    export("eel-habitat-harvester", roots, "Passive eel habitat harvester")
    print("   ", stats)
    roots, stats = scene_organ()
    export("artificial-electric-organ", roots, "Artificial electric organ (eel-inspired hydrogel stack)")
    print("   ", stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the 3D models for the energy-extraction design.

Two assemblies are built procedurally, in metres with +Y up:

  models/eel-habitat-harvester.*      Part A - passive habitat harvester (animated)
  models/artificial-electric-organ.*  Part B - eel-inspired hydrogel power cell (animated)

Each one is written as
  .glb        glTF 2.0 binary: materials, hierarchy, part descriptions in extras,
              a skinned eel and a looping animation clip
  .obj + .mtl Wavefront (one object per part, a single frozen moment)
  .stl        binary STL, millimetres, +Z up, opaque parts only (same moment)

Part A also writes models/eel-habitat-harvester.timeline.json: the physics behind
every frame (plate voltages, energies, stage windows) for the browser viewer.

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
FPS = 24


# --------------------------------------------------------------------------
# Mesh primitives
# --------------------------------------------------------------------------

class Mesh:
    def __init__(self, pos, nrm, idx, joints=None, weights=None):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        self.nrm = np.asarray(nrm, dtype=np.float64).reshape(-1, 3)
        self.idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        self.joints = None if joints is None else np.asarray(joints, dtype=np.int64).reshape(-1, 4)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1, 4)

    def xform(self, M):
        M = np.asarray(M, dtype=np.float64)
        pos = self.pos @ M[:3, :3].T + M[:3, 3]
        nrm = self.nrm @ np.linalg.inv(M[:3, :3])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        idx = self.idx[:, ::-1] if np.linalg.det(M[:3, :3]) < 0 else self.idx
        return Mesh(pos, nrm, idx, self.joints, self.weights)


def merge(meshes):
    pos, nrm, idx, off = [], [], [], 0
    for m in meshes:
        pos.append(m.pos)
        nrm.append(m.nrm)
        idx.append(m.idx + off)
        off += len(m.pos)
    skinned = all(m.joints is not None for m in meshes)
    return Mesh(np.vstack(pos), np.vstack(nrm), np.vstack(idx),
                np.vstack([m.joints for m in meshes]) if skinned else None,
                np.vstack([m.weights for m in meshes]) if skinned else None)


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


def mat_to_quat(m):
    """3x3 rotation -> glTF quaternion (x, y, z, w)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat4(q):
    x, y, z, w = q
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return M


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


def closed_spline(ctrl, per=60):
    """Closed uniform Catmull-Rom loop through the control points."""
    P = np.asarray(ctrl, float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


class Polyline:
    """Arc-length parameterised polyline (open or closed)."""

    def __init__(self, pts, closed=False):
        P = np.asarray(pts, float)
        if closed:
            P = np.vstack([P, P[:1]])
        self.P, self.closed = P, closed
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        self.length = float(self.s[-1])

    def at(self, u):
        u = np.asarray(u, float)
        u = np.mod(u, self.length) if self.closed else np.clip(u, 0, self.length)
        return np.stack([np.interp(u, self.s, self.P[:, k]) for k in range(3)], axis=-1)

    def tangent(self, u, h=0.01):
        d = self.at(np.asarray(u) + h) - self.at(np.asarray(u) - h)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


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
# Scene graph, skins, animation channels and materials
# --------------------------------------------------------------------------

class Node:
    """A glTF node. `prims` is a list of (Mesh, material); TRS is local.

    anim_only: the node exists for the animation (glows, energy packets) and is
               left out of the frozen OBJ / STL exports.
    static:    world-space prims to export to OBJ / STL instead (skinned meshes).
    """

    def __init__(self, name, mesh=None, mat=None, desc=None, children=None, stl=True,
                 prims=None, t=None, r=None, s=None, skin=None, static=None, anim_only=False):
        self.name, self.desc = name, desc
        self.prims = prims if prims is not None else ([(mesh, mat)] if mesh is not None else [])
        self.children = children or []
        self.stl, self.skin, self.static, self.anim_only = stl, skin, static, anim_only
        self.t, self.r, self.s = t, r, s

    def local(self):
        M = np.eye(4)
        if self.t is not None:
            M = M @ T(*self.t)
        if self.r is not None:
            M = M @ quat_to_mat4(self.r)
        if self.s is not None:
            M = M @ S(*self.s)
        return M

    def walk(self, M=None, prefix=""):
        W = (np.eye(4) if M is None else M) @ self.local()
        path = prefix + self.name
        yield path, self, W
        for c in self.children:
            yield from c.walk(W, path + "/")


class Skin:
    def __init__(self, joints, ibm):
        self.joints, self.ibm = joints, ibm


class Channel:
    def __init__(self, node, path, times, values, interp="LINEAR"):
        self.node, self.path, self.interp = node, path, interp
        self.times = np.asarray(times, float)
        self.values = np.asarray(values, float)

    def sample(self, t):
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        i = min(max(i, 0), len(self.times) - 1)
        if self.interp == "STEP" or i == len(self.times) - 1:
            return self.values[i]
        a, b = self.times[i], self.times[i + 1]
        f = 0.0 if b <= a else (t - a) / (b - a)
        v = self.values[i] * (1 - f) + self.values[i + 1] * f
        return v / np.linalg.norm(v) if self.path == "rotation" else v


def mat(color, alpha=1.0, metal=0.0, rough=0.5, emit=None, double=False, glow=1.0):
    return {"color": color, "alpha": alpha, "metal": metal, "rough": rough,
            "emit": emit or (0.0, 0.0, 0.0), "double": double, "glow": glow}


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
    "eel_skin": mat((0.24, 0.22, 0.17), rough=0.38),
    "eel_fin": mat((0.19, 0.16, 0.13), rough=0.5, double=True),
    "eel_eye": mat((0.02, 0.02, 0.02), rough=0.05),
    "food": mat((0.95, 0.56, 0.48), rough=0.5),
    "field_glow": mat((1.0, 0.70, 0.10), rough=0.4, emit=(1.0, 0.55, 0.02), glow=1.4),
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
    # Part A animation helpers
    "glow_plus": mat((1.0, 0.30, 0.10), alpha=0.85, rough=0.3, emit=(1.0, 0.22, 0.04), glow=1.6),
    "glow_minus": mat((0.20, 0.45, 1.0), alpha=0.85, rough=0.3, emit=(0.08, 0.30, 1.0), glow=1.6),
    "energy_packet": mat((1.0, 0.80, 0.20), rough=0.3, emit=(1.0, 0.70, 0.08), glow=2.0),
    "glow_component": mat((1.0, 0.78, 0.25), alpha=0.5, rough=0.2, emit=(1.0, 0.65, 0.10), glow=1.6),
    "beacon_flash": mat((1.0, 0.62, 0.15), alpha=0.75, rough=0.2, emit=(1.0, 0.45, 0.02), glow=3.0),
    "led_flash": mat((0.3, 1.0, 0.45), alpha=0.8, rough=0.2, emit=(0.1, 1.0, 0.3), glow=2.5),
    "epaper_flash": mat((0.02, 0.02, 0.02), rough=0.9),
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
    "ion_cation": mat((0.98, 0.48, 0.10), rough=0.4, emit=(0.6, 0.22, 0.0), glow=1.5),
    "ion_anion": mat((0.22, 0.72, 0.38), rough=0.4, emit=(0.08, 0.45, 0.15), glow=1.5),
    "current_arrow": mat((1.0, 0.86, 0.25), rough=0.4, emit=(0.6, 0.45, 0.05)),
    "pin_steel": mat((0.7, 0.7, 0.72), metal=1.0, rough=0.35),
}


def step_track(windows, period):
    """STEP scale keys that show a node inside the given (start, end) windows."""
    split = []
    for a, b in windows:              # the clip loops, so windows past the end wrap to the start
        if b > period:
            split += [(a, period), (0.0, b - period)]
        else:
            split.append((a, b))
    wins = sorted((max(0.0, a), min(period, b)) for a, b in split if b > 0 and a < period and b > a)
    merged = []
    for a, b in wins:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    times, vals = [0.0], [1.0 if merged and merged[0][0] <= 1e-6 else 0.0]

    def put(t, v):
        if abs(t - times[-1]) < 1e-6:
            vals[-1] = v
        else:
            times.append(t)
            vals.append(v)

    for a, b in merged:
        if a > 1e-6:
            put(a, 1.0)
        if b < period - 1e-6:
            put(b, 0.0)
    if times[-1] < period - 1e-6:
        put(period, vals[-1])
    return times, [[v, v, v] for v in vals]


# --------------------------------------------------------------------------
# Part A - passive eel habitat harvester
# --------------------------------------------------------------------------

G = 0.019                      # glass thickness
IN_X, IN_Z = 1.5 - G, 0.6 - G  # inner half-length / half-depth of the tank
Y_BOT = 0.80 + G               # inner floor (sand is saturated, so it conducts)
SUB_TOP = Y_BOT + 0.04         # top of the substrate
WATER_TOP = 1.68               # leaves a 12 cm air gap: eels breathe air
LID_Y = 1.806
EY = 1.27                      # electrode centre height
E0 = np.array([1.95, 1.05, -0.25])  # harvester enclosure origin (bottom centre)

# Eel and electrical model (see docs/DESIGN.md sections 2-3)
L_EEL = 1.65                   # body length used for the rig, m
NJ = 24                        # joints along the body
S_J = np.linspace(0, L_EEL, NJ)
MID_J = NJ // 2
SIGMA = 0.01                   # water conductivity, S/m (100 uS/cm)
I_EEL = 1.0                    # discharge current, A
V_ORGAN = 600.0                # organ voltage during a feeding volley, V
P_WATER = 300.0                # power the eel puts into the water per pulse, W
R_PAIR = 240.0                 # spreading resistance of a plate pair, ohm
T_PULSE = 0.002                # pulse width, s
ETA = 0.70                     # converter efficiency
DIODE = 0.7                    # one diode drop, V (two conduct)
K_PHI = I_EEL / (4 * math.pi * SIGMA)
LV_VOLTS = 10.0                # low-voltage sensing pulses
SC_C, SC_E0 = 1.0, 3.0         # supercapacitor bank: 1 F, 3 J at the start of the loop
PHONE_J = 15 * 3600.0
TYPICAL_DAY_J = 0.14

# Closed swimming loop for the head (front straight runs toward -x).
# The loop starts just before the hide, so the feeding strike falls mid-loop.
SWIM_PATH = [(-0.95, 1.19, -0.17), (-0.60, 1.17, -0.23), (-0.10, 1.21, -0.22), (0.40, 1.24, -0.20),
             (0.72, 1.25, -0.16), (1.02, 1.28, -0.04), (1.10, 1.30, 0.15), (0.95, 1.30, 0.28),
             (0.45, 1.36, 0.30), (0.08, 1.52, 0.30), (-0.15, 1.635, 0.28), (-0.45, 1.42, 0.30),
             (-0.85, 1.27, 0.26), (-1.08, 1.22, 0.05)]
CP_HIDE, CP_FOOD, CP_BREATH, CP_CORNER = 1, 4, 10, 13

# Visible flash patterns (seconds from event start). Real pulses are 2 ms at
# up to ~400 per second, far too fast to see, so the flashes are stretched.
DOUBLET_FLASHES = [(0.00, 0.08), (0.18, 0.26)]
VOLLEY_FLASHES = [(a, a + 0.06) for a in np.arange(0.0, 0.55, 0.1)] + \
                 [(a, a + 0.06) for a in np.arange(0.75, 1.30, 0.1)]
PACKET_TRAVEL = 1.1            # seconds for an energy packet to run along a lead (slowed for the eye)
PACKETS_PER_LEAD = 4


def eel_profile(t):
    """Radius and cross-section shape along the body, t = 0 (snout) .. 1 (tail)."""
    R0 = 0.05
    t = np.asarray(t, float)
    snout = np.sqrt(np.clip(1 - (1 - np.clip(t / 0.07, 0, 1)) ** 2, 0, 1))
    taper = np.where(t < 0.3, 1.0, np.clip((1 - t) / 0.7, 0, 1) ** 0.85)
    r = np.maximum(R0 * snout * taper, 0.0015)
    sx = np.interp(t, [0, 0.08, 0.25, 1.0], [1.22, 1.15, 1.0, 0.6])   # half-width factor
    sy = np.interp(t, [0, 0.08, 0.25, 1.0], [0.80, 0.86, 1.0, 1.3])   # half-height factor
    return r, sx, sy


def skin_weights(mesh):
    """Two-joint linear blend by distance along the straight bind pose (-x)."""
    s = np.clip(-mesh.pos[:, 0], 0, L_EEL)
    f = s / (L_EEL / (NJ - 1))
    i0 = np.clip(np.floor(f), 0, NJ - 2).astype(int)
    w1 = np.clip(f - i0, 0, 1)
    joints = np.zeros((len(s), 4), int)
    weights = np.zeros((len(s), 4))
    joints[:, 0], joints[:, 1] = i0, i0 + 1
    weights[:, 0], weights[:, 1] = 1 - w1, w1
    joints[weights == 0] = 0          # glTF: unused influences must point at joint 0
    mesh.joints, mesh.weights = joints, weights
    return mesh


def build_eel_bind():
    """Straight eel in its bind pose: snout at the origin, body along -x, up +y."""
    n = 240
    s = np.linspace(0, L_EEL, n)
    t = s / L_EEL
    P = np.stack([-s, np.zeros(n), np.zeros(n)], axis=1)
    r, sx, sy = eel_profile(t)
    k = 28
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    prof = np.stack([np.cos(ang)[None] * (r * sx)[:, None], np.sin(ang)[None] * (r * sy)[:, None]], axis=-1)
    body = sweep(P, prof, world_up=True)

    # Anal fin: the long ventral ribbon knifefish swim with.
    sel = t >= 0.12
    tf, Pf, hf = t[sel], P[sel], (r * sy)[sel]
    f = 0.03 * smoothstep(0.12, 0.26, tf) * (0.35 + 0.65 * (1 - smoothstep(0.85, 1.0, tf)))
    th = 0.0018
    top, bot = -hf * 0.7, -hf - f
    one = np.ones_like(tf)
    fin_prof = np.stack([np.stack([-th * one, top], 1), np.stack([th * one, top], 1),
                         np.stack([th * one, bot], 1), np.stack([-th * one, bot], 1)], axis=1)
    fin = sweep(Pf, fin_prof, world_up=True)

    i = int(0.045 * (n - 1))
    eyes = [sphere(0.0065, 16, 10).xform(T(P[i, 0], r[i] * sy[i] * 0.42, side * r[i] * sx[i] * 0.86))
            for side in (-1, 1)]
    return [skin_weights(body), skin_weights(fin), skin_weights(merge(eyes))]


def skin_cpu(mesh, joint_world, ibm):
    """Pose a skinned mesh on the CPU (for the frozen OBJ / STL exports)."""
    Mj = joint_world @ ibm
    Mv = np.einsum("nk,nkij->nij", mesh.weights, Mj[mesh.joints])
    pos = np.einsum("nij,nj->ni", Mv[:, :3, :3], mesh.pos) + Mv[:, :3, 3]
    nrm = np.einsum("nij,nj->ni", Mv[:, :3, :3], mesh.nrm)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return Mesh(pos, nrm, mesh.idx)


class EelMotion:
    """Follow-the-leader swimming along a closed loop, plus a travelling body wave."""

    def __init__(self):
        self.loop = Polyline(closed_spline(SWIM_PATH, 60), closed=True)
        L = self.loop.length
        cp = self.loop.P[:-1]
        self.cp_u = [float(self.loop.s[np.argmin(np.linalg.norm(cp - np.asarray(c), axis=1))]) for c in SWIM_PATH]
        u = np.linspace(0, L, 4001)

        def bump(c, w):
            d = (u - c + L / 2) % L - L / 2
            return np.exp(-(d / w) ** 2)

        v = 0.18 * (1 - 0.45 * bump(self.cp_u[CP_BREATH], 0.45)) * (1 + 1.3 * bump(self.cp_u[CP_FOOD] - 0.12, 0.30))
        tt = np.concatenate([[0.0], np.cumsum(np.diff(u) * 0.5 * (1 / v[1:] + 1 / v[:-1]))])
        self.frames = int(round(tt[-1] * FPS))
        self.T = self.frames / FPS
        self.u, self.t_of_u = u, tt * (self.T / tt[-1])
        self.v = v * tt[-1] / self.T
        self.freq = round(0.9 * self.T) / self.T    # whole number of body waves per loop
        self.lam = 0.95

    def time_at(self, cp_index):
        return float(np.interp(self.cp_u[cp_index], self.u, self.t_of_u))

    def speed(self, tau):
        return float(np.interp(np.interp(tau % self.T, self.t_of_u, self.u), self.u, self.v))

    def pose(self, tau):
        """World positions (NJ, 3) and rotations (NJ, 3, 3) of the joints."""
        sh = float(np.interp(tau % self.T, self.t_of_u, self.u))
        uj = sh - S_J
        base = self.loop.at(uj)
        tan = self.loop.tangent(uj)
        side = np.cross(np.array([0.0, 1.0, 0.0]), tan)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
        amp = 0.012 + 0.05 * (S_J / L_EEL) ** 1.6
        P = base + side * (amp * np.sin(2 * np.pi * (S_J / self.lam - self.freq * tau)))[:, None]
        fwd = np.empty_like(P)
        fwd[1:-1] = P[:-2] - P[2:]
        fwd[0], fwd[-1] = P[0] - P[1], P[-2] - P[-1]
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
        up = np.array([0.0, 1.0, 0.0]) - fwd * fwd[:, 1:2]
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        Rm = np.stack([fwd, up, np.cross(fwd, up)], axis=2)
        return P, Rm

    def body_point(self, P, s):
        return np.stack([np.interp(s, S_J, P[:, k]) for k in range(3)])


def _images(p):
    """The point plus its mirror images in the six insulating tank boundaries."""
    x, y, z = p
    return np.array([p, (-2 * IN_X - x, y, z), (2 * IN_X - x, y, z), (x, y, -2 * IN_Z - z), (x, y, 2 * IN_Z - z),
                     (x, 2 * Y_BOT - y, z), (x, 2 * WATER_TOP - y, z)])


def potential(points, H, Tl, volts=V_ORGAN):
    """Potential (V) at points from a head(+)/tail(-) current dipole in an insulated tank."""
    k = K_PHI * volts / V_ORGAN
    pts = np.asarray(points, float)
    phi = np.zeros(len(pts))
    for hi, ti in zip(_images(H), _images(Tl)):
        phi += 1 / np.linalg.norm(pts - hi, axis=1) - 1 / np.linalg.norm(pts - ti, axis=1)
    return k * phi


def efield(p, H, Tl):
    v = np.zeros(3)
    for hi, ti in zip(_images(H), _images(Tl)):
        a, b = p - hi, p - ti
        v += a / np.linalg.norm(a) ** 3 - b / np.linalg.norm(b) ** 3
    return v / np.linalg.norm(v)


def field_lines(H, Tl, body_P, body_r, max_lines=10):
    """Head(+) to tail(-) field lines that stay in the water and clear the body."""
    lo = np.array([-IN_X + 0.06, SUB_TOP + 0.03, -IN_Z + 0.09])
    hi = np.array([IN_X - 0.06, WATER_TOP - 0.02, IN_Z - 0.03])
    golden = math.pi * (3 - math.sqrt(5))
    lines = []
    for j in range(140):
        yv = 1 - 2 * (j + 0.5) / 140
        rad = math.sqrt(1 - yv * yv)
        d = np.array([math.cos(golden * j) * rad, yv, math.sin(golden * j) * rad])
        p = H + 0.07 * d
        pts, ok = [p.copy()], False
        for _ in range(1400):
            p = p + 0.007 * efield(p + 0.0035 * efield(p, H, Tl), H, Tl)
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
        if np.any(dist - body_r[None] < 0.025):
            continue
        axis = (Tl - H) / np.linalg.norm(Tl - H)
        reach = float(np.max(np.linalg.norm(np.cross(pts - H, axis), axis=1)))
        lines.append((reach, pts))
    lines.sort(key=lambda x: x[0])
    if len(lines) > max_lines:
        pick = np.linspace(0, len(lines) - 1, max_lines).round().astype(int)
        lines = [lines[i] for i in sorted(set(pick))]
    return [resample(pts, 0.015) for _, pts in lines]


def build_electrodes():
    """Six graphite plates, each behind a perforated acrylic guard."""
    specs = [("E1", (-IN_X + 0.016, EY, 0.0), 90)]
    specs += [(f"E{i + 2}", (xp, EY, -IN_Z + 0.016), 0) for i, xp in enumerate((-1.05, -0.35, 0.35, 1.05))]
    specs += [("E6", (IN_X - 0.016, EY, 0.0), -90)]
    PW, PH, PT = 0.30, 0.45, 0.008
    plates, guards, stand, tops, frames_ = [], [], [], [], []
    for name, c, rot in specs:
        M = T(*c) @ R("y", rot)
        plates.append(box((PW, PH, PT)).xform(M))
        for sx_ in (-1, 1):
            for sy_ in (-1, 1):
                stand.append(cylinder(0.006, 0.012, 12).xform(M @ T(sx_ * 0.12, sy_ * 0.19, -0.010) @ R("x", 90)))
                stand.append(cylinder(0.007, 0.072, 12).xform(M @ T(sx_ * 0.18, sy_ * 0.255, 0.020) @ R("x", 90)))
        guards.append(lattice(0.38, 0.53, 0.006).xform(M @ T(0, 0, 0.059)))
        tops.append((name, (M @ np.array([0, PH / 2, 0, 1]))[:3], rot))
        frames_.append((name, M))
    return merge(plates), merge(guards), merge(stand), tops, frames_


def plate_samples(M, n=5):
    g = np.linspace(-1, 1, n)
    pts = [(M @ np.array([0.13 * a, 0.20 * b, 0.004, 1.0]))[:3] for a in g for b in g]
    return np.array(pts)


def electrode_cable_paths(tops):
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
    return paths


PCB_Y = 0.0208
HARVESTER_GLOWS = {   # enclosure-local boxes (size, centre) lit as energy moves through
    "Bridge glow": ((0.064, 0.016, 0.232), (-0.11, PCB_Y + 0.006, 0.0)),
    "Catch capacitor glow": ((0.066, 0.044, 0.030), (0.0, PCB_Y + 0.018, -0.07)),
    "Converter glow": ((0.058, 0.020, 0.043), (0.055, PCB_Y + 0.008, 0.05)),
    "Supercapacitor glow": ((0.036, 0.042, 0.066), (0.10, PCB_Y + 0.0175, -0.07)),
}


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
    parts["Main board"] = (box((0.38, 0.0016, 0.28), (0, PCB_Y - 0.0008, 0)).xform(M), "pcb_green")
    term_in = box((0.02, 0.016, 0.24), (-0.17, PCB_Y + 0.008, 0))
    term_out = box((0.02, 0.016, 0.09), (0.17, PCB_Y + 0.008, 0.08))
    parts["Terminal blocks"] = (merge([term_in, term_out]).xform(M), "terminal_green")
    screws = [cylinder(0.004, 0.003, 12).xform(T(-0.17, PCB_Y + 0.0175, -0.10 + 0.04 * i)) for i in range(6)]
    screws += [cylinder(0.004, 0.003, 12).xform(T(0.17, PCB_Y + 0.0175, 0.05 + 0.03 * i)) for i in range(3)]
    parts["Terminal screws"] = (merge(screws).xform(M), "screw_metal")
    # 6-input polyphase bridge: two diodes per plate, one to each rail.
    diodes = []
    for i in range(6):
        z = -0.10 + 0.04 * i
        for x in (-0.125, -0.095):
            diodes.append(cylinder(0.0035, 0.014, 12).xform(T(x, PCB_Y + 0.0035, z) @ R("z", 90)))
    parts["Polyphase bridge (12 diodes)"] = (merge(diodes).xform(M), "component_black")
    movs = [cylinder(0.012, 0.005, 24).xform(T(-0.055, PCB_Y + 0.013, z) @ R("x", 90)) for z in (-0.05, 0.0)]
    parts["Surge clamp (MOV + TVS)"] = (merge(movs).xform(M), "mov_blue")
    parts["Pulse catch capacitor"] = (box((0.058, 0.036, 0.022), (0.0, PCB_Y + 0.018, -0.07)).xform(M), "film_cap_yellow")
    parts["Buck converter board"] = (box((0.05, 0.002, 0.035), (0.055, PCB_Y + 0.006, 0.05)).xform(M), "pcb_blue")
    buck = [cylinder(0.009, 0.008, 24).xform(T(0.045, PCB_Y + 0.011, 0.05)),
            box((0.008, 0.002, 0.008), (0.068, PCB_Y + 0.008, 0.045)),
            box((0.016, 0.002, 0.016), (0.135, PCB_Y + 0.001, 0.06)),
            box((0.012, 0.004, 0.008), (0.10, PCB_Y + 0.002, -0.02))]
    parts["Converter, MCU and radio"] = (merge(buck).xform(M), "component_black")
    caps = [cylinder(0.0125, 0.035, 24).xform(T(0.10, PCB_Y + 0.0175, z)) for z in (-0.085, -0.055)]
    parts["Supercapacitors (2 x 2 F, series)"] = (merge(caps).xform(M), "supercap_sleeve")
    glands = [cylinder(0.008, 0.024, 16).xform(T(-0.222, 0.07, -0.10 + 0.04 * i) @ R("z", 90)) for i in range(6)]
    glands += [cylinder(0.008, 0.024, 16).xform(T(x, 0.07, 0.172) @ R("x", 90)) for x in (0.06, 0.11, 0.16)]
    parts["Cable glands"] = (merge(glands).xform(M), "component_black")
    return parts


LID_TOP = LID_Y + 0.006
BEACON_C = np.array([1.40, LID_TOP + 0.05, -0.575])
LED_C = np.array([1.17, LID_TOP + 0.03, -0.549])
DISPLAY_M = T(1.05, 1.17, 1.04) @ R("x", -15)


def _spikes(heights, x0, dx, base_y):
    return [box((0.003, h, 0.001), (x0 + dx * i, base_y + h / 2, 0.0125)) for i, h in enumerate(heights)]


def build_loads():
    out = {}
    out["Health monitor node"] = (box((0.10, 0.045, 0.05), (1.20, LID_TOP + 0.0225, -0.575)), "sensor_white")
    out["Radio antenna"] = (cylinder(0.004, 0.07, 12).xform(T(1.235, LID_TOP + 0.045 + 0.035, -0.575)), "component_black")
    out["Status LED"] = (sphere(0.004, 12, 8).xform(T(*LED_C)), "led_green")
    out["Probe guard"] = (pipe(0.022, 0.025, 0.22, 24).xform(T(1.25, 1.40, -0.40)), "guard_acrylic")
    out["Temperature + conductivity probe"] = (merge([
        cylinder(0.011, 0.20, 20).xform(T(1.25, 1.41, -0.40)),
        cylinder(0.006, 0.02, 16).xform(T(1.25, 1.30, -0.40)),
    ]), "probe_steel")
    out["Alarm beacon base"] = (cylinder(0.03, 0.025, 24).xform(T(1.40, LID_TOP + 0.0125, -0.575)), "component_black")
    out["Alarm beacon lens"] = (merge([
        cylinder(0.026, 0.03, 32, caps=False).xform(T(1.40, LID_TOP + 0.040, -0.575)),
        sphere(0.026, 32, 12, hemi=True).xform(T(1.40, LID_TOP + 0.055, -0.575)),
    ]), "beacon_amber")
    DX, DZ = 1.05, 1.02
    out["Display stand"] = (merge([
        cylinder(0.16, 0.015, 40).xform(T(DX, 0.0075, DZ)),
        cylinder(0.018, 1.02, 20).xform(T(DX, 0.525, DZ)),
        box((0.08, 0.05, 0.05), (DX, 1.06, DZ - 0.01)),
    ]), "display_black")
    P = DISPLAY_M
    out["Display frame"] = (box((0.46, 0.32, 0.02)).xform(P), "display_black")
    out["E-paper screen"] = (box((0.42, 0.28, 0.003), (0, 0, 0.0105)).xform(P), "epaper")
    ink = [box((0.36, 0.002, 0.001), (0, 0.045, 0.0125)), box((0.36, 0.002, 0.001), (0, -0.115, 0.0125))]
    for i, hgt in enumerate((0.035, 0.06, 0.028, 0.075, 0.05, 0.066)):
        ink.append(box((0.03, hgt, 0.001), (-0.15 + 0.05 * i, -0.114 + hgt / 2, 0.0125)))
    out["E-paper axes and history"] = (merge(ink).xform(P), "epaper_ink")
    return out


def load_cable_paths():
    g = lambda x: E0 + np.array([x, 0.07, 0.184])  # noqa: E731 - output gland tip
    beacon = [g(0.06), g(0.06) + (0, 0, 0.03), (1.95, 1.55, -0.03), (1.62, 1.90, -0.30), (1.47, 1.90, -0.52), (1.43, LID_TOP + 0.013, -0.575)]
    node = [g(0.11), g(0.11) + (0, 0, 0.05), (1.99, 1.58, -0.01), (1.62, 1.93, -0.27), (1.30, 1.90, -0.50), (1.25, LID_TOP + 0.02, -0.575)]
    probe = [(1.25, 1.51, -0.40), (1.25, 1.75, -0.40), (1.25, 1.84, -0.47), (1.22, LID_TOP + 0.045, -0.565)]
    disp = [g(0.16), g(0.16) + (0, 0, 0.06), (E0[0] + 0.16, 0.02, -0.01), (2.0, 0.006, 0.40), (1.17, 0.006, 0.97)]
    return [rounded_path([np.asarray(p, float) for p in pts], radius=0.08) for pts in (beacon, node, probe, disp)]


def static_habitat():
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
    return [cab, plinth, handles, glass, frame, water, sand, rock, hide, lid]


def packet_keys(line, t_start, t_end, k, forward, period):
    """LINEAR translation keys for one energy packet streaming along a lead."""
    def pos(f):
        return line.at((f if forward else 1 - f) * line.length)

    phase0 = k / PACKETS_PER_LEAD
    times = list(np.arange(t_start, t_end, 1 / 30)) + [t_end]
    keys = []
    for i, t in enumerate(times):
        ph = (t - t_start) / PACKET_TRAVEL + phase0
        if i:
            prev = (times[i - 1] - t_start) / PACKET_TRAVEL + phase0
            if math.floor(ph) > math.floor(prev):   # wrap: jump back to the start of the lead
                tw = t_start + (math.floor(ph) - phase0) * PACKET_TRAVEL
                keys.append((tw - 1e-3, pos(1.0)))
                keys.append((tw, pos(0.0)))
        keys.append((t, pos(ph % 1.0)))
    out = []
    for t, p in keys:
        if 0 <= t <= period and (not out or t > out[-1][0] + 1e-5):
            out.append((t, p))
    return out


def scene_habitat():
    stats = {}
    motion = EelMotion()
    period = motion.T

    # ---- events: what happens during one loop ---------------------------------
    t_corner = motion.time_at(CP_CORNER)
    t_hide = motion.time_at(CP_HIDE)
    t_food = motion.time_at(CP_FOOD)
    t_breath = motion.time_at(CP_BREATH)
    hv_specs = [
        dict(id="probe_corner", kind="probe", t0=t_corner - 0.15, flashes=DOUBLET_FLASHES, pulses=2, rate=250,
             beacon=3.0, label="Probing doublet near the end wall",
             story="Two quick high-voltage pulses to check the corner. The head is close to plate E1."),
        dict(id="probe_hide", kind="probe", t0=t_hide - 0.15, flashes=DOUBLET_FLASHES, pulses=2, rate=250,
             beacon=3.0, label="Probing doublet over the hide",
             story="Electric eels fire doublets to make hidden prey twitch. Mid-tank, so the plates see less."),
        dict(id="feed", kind="feed", t0=t_food - 0.25, flashes=VOLLEY_FLASHES, pulses=70, rate=330,
             beacon=5.0, label="Feeding strike: 70-pulse volley",
             story="The eel strikes the food with two bursts of high-voltage pulses, close to the back-wall plates."),
    ]

    plates, guards, stand, tops, plate_frames = build_electrodes()
    plate_pts = [plate_samples(M) for _, M in plate_frames]
    plate_names = [n for n, _ in plate_frames]
    lead_paths = electrode_cable_paths(tops)
    leads = [Polyline(p) for p in lead_paths]

    events = []
    for spec in hv_specs:
        vis = max(b for _, b in spec["flashes"])
        tm = spec["t0"] + vis / 2
        P, _ = motion.pose(tm)
        H = motion.body_point(P, 0.05 * L_EEL)
        Tl = motion.body_point(P, 0.97 * L_EEL)
        phis = np.array([potential(pts, H, Tl).mean() for pts in plate_pts])
        ip, im = int(np.argmax(phis)), int(np.argmin(phis))
        dV = float(phis[ip] - phis[im])
        rail = max(dV - 2 * DIODE, 0.0)
        p_w = rail ** 2 / (4 * R_PAIR)
        e_pulse = p_w * T_PULSE
        ev = dict(spec, vis=vis, t_mid=tm, plates_V=[round(float(v), 2) for v in phis], plus=ip, minus=im,
                  dV=round(dV, 2), rail=round(rail, 2), cap_V=round(rail / 2, 2), power_W=round(p_w, 3),
                  mJ_pulse=round(e_pulse * 1e3, 3), mJ_event=round(e_pulse * spec["pulses"] * 1e3, 2),
                  mJ_stored=round(e_pulse * spec["pulses"] * ETA * 1e3, 2),
                  load_pct=round(100 * p_w / P_WATER, 3), organ_V=V_ORGAN,
                  head=[round(float(v), 3) for v in H], tail=[round(float(v), 3) for v in Tl])
        ev["stages"] = {"discharge": [0, vis], "field": [0, vis], "plates": [0, vis + 0.25],
                        "bridge": [0.3, vis + 1.6], "store": [0.9, vis + 2.4], "loads": [0.5, vis + spec["beacon"]]}
        events.append(ev)
    feed = events[-1]
    t_refresh = feed["t0"] + feed["vis"] + 2.6
    t_drop = t_food - 4.2

    # Sensing pulses are too weak to harvest: show why with numbers.
    P0, _ = motion.pose(0.0)
    lv_phis = np.array([potential(pts, motion.body_point(P0, 0.05 * L_EEL), motion.body_point(P0, 0.97 * L_EEL),
                                  volts=LV_VOLTS).mean() for pts in plate_pts])
    lv_dV = float(lv_phis.max() - lv_phis.min())

    # ---- rig and skinned eel ----------------------------------------------------
    rest_t = feed["t0"] + 0.42          # frozen moment: mid-volley, a flash is on
    ibm = np.array([T(s, 0, 0) for s in S_J])
    joints = [Node(f"eel_joint_{j:02d}") for j in range(NJ)]
    rig = Node("Eel rig", children=joints, desc="24-joint swimming rig. The body follows the head's path.")
    skin = Skin(joints, ibm)
    body, fin, eyes = build_eel_bind()
    eel = Node("Electric eel", prims=[(body, "eel_skin"), (fin, "eel_fin"), (eyes, "eel_eye")], skin=skin,
               desc="Electrophorus, about 1.65 m. Its electric organ fills roughly 80% of the body: about 6,000 "
                    "electrocytes in series, ~0.15 V each, up to 860 V (E. voltai) for ~2 ms. It swims, surfaces "
                    "to breathe and hunts normally; the machine never stimulates, baits or restrains it.")
    channels = []
    ft = np.arange(motion.frames + 1) / FPS
    poses = [motion.pose(t) for t in ft]
    for j, jn in enumerate(joints):
        tr = np.array([p[0][j] for p in poses])
        qs = np.array([mat_to_quat(p[1][j]) for p in poses])
        for k in range(1, len(qs)):
            if np.dot(qs[k], qs[k - 1]) < 0:
                qs[k] = -qs[k]
        channels.append(Channel(jn, "translation", ft, tr))
        channels.append(Channel(jn, "rotation", ft, qs))

    # Field lines for each high-voltage event, attached to the middle joint so they ride with the eel.
    field_nodes = []
    for ev in events:
        P, Rm = motion.pose(ev["t_mid"])
        r_body = eel_profile(S_J / L_EEL)[0] * 1.3
        lines = field_lines(np.array(ev["head"]), np.array(ev["tail"]), P[3:-2], r_body[3:-2])
        ev["field_lines"] = len(lines)
        Mmid = np.eye(4)
        Mmid[:3, :3], Mmid[:3, 3] = Rm[MID_J], P[MID_J]
        inv = np.linalg.inv(Mmid)
        mesh = merge([tube(l, 0.0024, 6) for l in lines]).xform(inv)
        fn = Node(f"Field lines ({ev['id']})", mesh, "field_glow",
                  "Head-positive dipole field of this discharge, computed with the tank walls as insulating "
                  "mirrors. Most of the energy heats the water near the eel; the plates intercept about 0.1-1%.",
                  stl=False, anim_only=ev is not feed)
        field_nodes.append(fn)
        channels.append(Channel(fn, "scale", *step_track([(ev["t0"] + a, ev["t0"] + b) for a, b in ev["flashes"]],
                                                         period), interp="STEP"))
    joints[MID_J].children = field_nodes

    # ---- energy flow overlays -------------------------------------------------------
    flow = []
    for i, (pname, M) in enumerate(plate_frames):
        for pol, matname, key in (("+", "glow_plus", "plus"), ("-", "glow_minus", "minus")):
            wins = [(ev["t0"], ev["t0"] + ev["vis"] + 0.25) for ev in events if ev[key] == i]
            if not wins:
                continue
            n = Node(f"Plate {pname} glow ({pol})", box((0.31, 0.46, 0.004), (0, 0, 0.0065)).xform(M), matname,
                     f"Lights when plate {pname} is the {'positive' if pol == '+' else 'negative'} plate of the "
                     "conducting pair.", stl=False, anim_only=True)
            channels.append(Channel(n, "scale", *step_track(wins, period), interp="STEP"))
            flow.append(n)
    packet_mesh = sphere(0.011, 12, 8)
    for c, line in enumerate(leads):
        evs = [ev for ev in events if c in (ev["plus"], ev["minus"])]
        if not evs:
            continue
        for k in range(PACKETS_PER_LEAD):
            keys = []
            for ev in evs:
                keys += packet_keys(line, ev["t0"] + 0.05, ev["t0"] + ev["vis"] + 1.15, k, c == ev["plus"], period)
            keys.sort(key=lambda x: x[0])
            if keys[0][0] > 0:
                keys.insert(0, (0.0, keys[0][1]))
            if keys[-1][0] < period:
                keys.append((period, keys[-1][1]))
            pn = Node(f"Energy packet {plate_names[c]}-{k}", packet_mesh, "energy_packet",
                      "Marks current in this lead: toward the harvester on the + lead, back to the plate on the "
                      "- lead. Real current is instant; the dots are slowed so you can follow the path.",
                      stl=False, anim_only=True, t=tuple(keys[0][1]))
            channels.append(Channel(pn, "translation", [t for t, _ in keys], [p for _, p in keys]))
            wins = [(ev["t0"] + 0.05, ev["t0"] + ev["vis"] + 1.15) for ev in evs]
            channels.append(Channel(pn, "scale", *step_track(wins, period), interp="STEP"))
            flow.append(pn)
    offsets = {"Bridge glow": (0.3, 1.2), "Catch capacitor glow": (0.4, 1.6), "Converter glow": (0.6, 1.9),
               "Supercapacitor glow": (0.9, 2.4)}
    for name, (size, centre) in HARVESTER_GLOWS.items():
        a, b = offsets[name]
        n = Node(name, box(size, centre).xform(T(*E0)), "glow_component", stl=False, anim_only=True)
        channels.append(Channel(n, "scale", *step_track([(ev["t0"] + a, ev["t0"] + ev["vis"] + b) for ev in events],
                                                        period), interp="STEP"))
        flow.append(n)
    beacon_w, led_w = [], []
    for ev in events:
        for a in np.arange(0.5, ev["vis"] + ev["beacon"], 0.5):
            beacon_w.append((ev["t0"] + a, ev["t0"] + a + 0.25))
        led_w += [(ev["t0"] + a, ev["t0"] + a + 0.15) for a in (0.6, 0.9, 1.2)]
    bn = Node("Beacon flash", sphere(0.034, 20, 12).xform(T(*BEACON_C)), "beacon_flash",
              "The alarm flashes after every high-voltage discharge, powered by the discharge itself.",
              stl=False, anim_only=True)
    ln = Node("Monitor LED flash", sphere(0.009, 12, 8).xform(T(*LED_C)), "led_flash",
              "The health monitor logs every discharge it detects.", stl=False, anim_only=True)
    channels.append(Channel(bn, "scale", *step_track(beacon_w, period), interp="STEP"))
    channels.append(Channel(ln, "scale", *step_track(led_w, period), interp="STEP"))
    flow += [bn, ln]

    # Display: old waveform until it refreshes after the feeding volley, then the new one.
    wave_a = merge(_spikes((0.07, 0.0, 0.0, 0.07), -0.15, 0.012, 0.045)
                   + _spikes((0.012,) * 5, 0.03, 0.03, 0.045)).xform(DISPLAY_M)
    burst = tuple(0.075 * 0.93 ** i for i in range(6))
    wave_b = merge(_spikes(burst, -0.16, 0.009, 0.045) + _spikes(tuple(0.06 * 0.9 ** i for i in range(5)), -0.08, 0.009, 0.045)
                   + _spikes((0.012,) * 4, 0.05, 0.03, 0.045)).xform(DISPLAY_M)
    new_bar = box((0.03, 0.058, 0.001), (0.15, -0.114 + 0.029, 0.0125)).xform(DISPLAY_M)
    flash = box((0.42, 0.28, 0.002), (0, 0, 0.0128)).xform(DISPLAY_M)
    wa = Node("E-paper waveform (before)", wave_a, "epaper_ink", stl=False)
    wb = Node("E-paper waveform (after refresh)", wave_b, "epaper_ink", stl=False, anim_only=True)
    nb = Node("E-paper new bar", new_bar, "epaper_ink", stl=False, anim_only=True)
    fl = Node("E-paper refresh flash", flash, "epaper_flash", stl=False, anim_only=True)
    channels.append(Channel(wa, "scale", *step_track([(0, t_refresh + 0.2)], period), interp="STEP"))
    channels.append(Channel(wb, "scale", *step_track([(t_refresh + 0.2, period)], period), interp="STEP"))
    channels.append(Channel(nb, "scale", *step_track([(t_refresh + 0.2, period)], period), interp="STEP"))
    channels.append(Channel(fl, "scale", *step_track([(t_refresh, t_refresh + 0.35)], period), interp="STEP"))

    # Food: dropped through the hatch, sinks, eaten at the strike.
    Pf, Rf = motion.pose(t_food)
    food_end = Pf[0] + Rf[0][:, 0] * 0.03
    food_start = np.array([food_end[0] - 0.02, WATER_TOP - 0.02, food_end[2] + 0.02])
    food_mesh = sphere(1.0, 14, 10).xform(S(0.034, 0.016, 0.016))
    food = Node("Food (shrimp)", food_mesh, "food", "Dropped through the feeding hatch by a keeper.",
                stl=False, anim_only=True, t=tuple(food_start))
    channels.append(Channel(food, "translation", [0, t_drop, t_food, period],
                            [food_start, food_start, food_end, food_end]))
    channels.append(Channel(food, "scale", *step_track([(t_drop, t_food + 0.12)], period), interp="STEP"))

    # ---- rest pose = the frozen moment used by OBJ/STL and non-animating viewers ---
    for ch in channels:
        setattr(ch.node, {"translation": "t", "rotation": "r", "scale": "s"}[ch.path], tuple(ch.sample(rest_t)))
    joint_world = np.array([jn.local() for jn in joints])
    eel.static = [(skin_cpu(m, joint_world, ibm), mt) for m, mt in eel.prims]

    # ---- assemble -------------------------------------------------------------------
    habitat = Node("Habitat", children=static_habitat() + [food],
                   desc="Welfare comes first: a very large tank, soft warm water, places to hide.")
    electrodes = Node("Electrode array", children=[
        Node("Graphite plates E1-E6", plates, "graphite", "Six inert graphite plates, 300 x 450 mm. No copper or "
             "zinc: their ions are toxic to fish."),
        Node("Guard screens", guards, "guard_acrylic", "Perforated acrylic, 75 mm from the glass. The eel can never "
             "touch an electrode."),
        Node("Standoffs", stand, "standoff_pvc"),
    ], desc="Six plates feed a polyphase bridge, so whichever two plates see the biggest difference conduct, "
            "in either polarity, wherever the eel points.")
    cables = Node("Cabling", children=[
        Node("Electrode leads", merge([tube(p, 0.0045, 10) for p in lead_paths]), "cable",
             "One insulated lead per plate, over the rim to the harvester's cable glands."),
        Node("Load cables", merge([tube(p, 0.0035, 10) for p in load_cable_paths()]), "cable"),
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
        "Education display": ["Display stand", "Display frame", "E-paper screen", "E-paper axes and history"],
    }
    gdesc = {
        "Eel health monitor": "Logs water temperature, conductivity and every discharge the plates detect; a change "
                              "in discharge pattern is an early sign of stress or illness. Runs on ~25 uW from its "
                              "own 10-year lithium cell, so it never depends on a sick eel discharging.",
        "Keeper shock alarm": "Every high-voltage discharge flashes the beacon: tank is live, hands out. Powered by "
                              "the very shock it warns about.",
        "Education display": "E-paper panel: energy harvested today, the last pulse waveform, and how long one eel "
                             "would take to charge a phone (about a thousand years). It refreshes only when the "
                             "eel has made enough surplus energy.",
    }
    load_nodes = [Node(g, children=[Node(p, lparts[p][0], lparts[p][1]) for p in pl], desc=gdesc[g])
                  for g, pl in groups.items()]
    load_nodes[2].children += [wa, wb, nb, fl]
    loads = Node("Loads (right purposes)", children=load_nodes)
    energy = Node("Energy flow (animated)", children=flow,
                  desc="Glows and moving dots that show where the energy goes during each discharge.")

    # ---- timeline for the viewer's panels --------------------------------------------
    timeline = build_timeline(motion, events, t_drop, t_food, t_breath, t_refresh, plate_names, lv_dV, rest_t)
    stats.update(loop_s=round(period, 2), eel_body_m=L_EEL,
                 events={ev["id"]: (plate_names[ev["plus"]], plate_names[ev["minus"]], ev["dV"], ev["mJ_pulse"],
                                    ev["mJ_event"], ev["field_lines"]) for ev in events},
                 lv_dV=round(lv_dV, 3))
    roots = [habitat, eel, rig, electrodes, cables, harvester, loads, energy]
    return roots, channels, timeline, stats


def build_timeline(motion, events, t_drop, t_food, t_breath, t_refresh, plate_names, lv_dV, rest_t):
    period = motion.T
    ts = np.round(np.arange(0, period + 1e-9, 0.05), 3)

    def ramp(t, a, b):
        return float(np.clip((t - a) / (b - a), 0, 1))

    def cap_v(t):
        v = 0.0
        for ev in events:
            for shift in (0.0, -period):          # the previous loop's tail wraps round
                d = t - (ev["t0"] + shift)
                if d < 0:
                    continue
                vis, top = ev["vis"], ev["cap_V"]
                if d <= vis:
                    x = top * (1 - math.exp(-d / 0.05))
                else:
                    x = top * math.exp(-(d - vis) / 0.6)
                    if x < 6.0:
                        t_sleep = vis + 0.6 * math.log(top / 6.0) if top > 6 else vis
                        x = min(top, 6.0) * math.exp(-(d - t_sleep) / 8.0)
                v = max(v, x)
        return v

    series = {"t": ts.tolist(), "cap_V": [], "harvest_mJ": [], "stored_mJ": [], "speed": [], "lv_hz": []}
    for t in ts:
        series["cap_V"].append(round(cap_v(t), 2))
        series["harvest_mJ"].append(round(sum(ev["mJ_event"] * ramp(t, ev["t0"] + 0.3, ev["t0"] + ev["vis"] + 1.2)
                                              for ev in events), 3))
        series["stored_mJ"].append(round(sum(ev["mJ_stored"] * ramp(t, ev["t0"] + 0.9, ev["t0"] + ev["vis"] + 2.4)
                                             for ev in events), 3))
        series["speed"].append(round(motion.speed(t), 3))
        busy = sum(math.exp(-((t - c) / 1.6) ** 2) for c in (events[0]["t0"], events[1]["t0"], t_food))
        series["lv_hz"].append(round(6 + 34 * min(busy, 1.0), 1))

    captions = [
        dict(t0=0.0, text="Cruising and sensing with weak ~10 V pulses. They are far too weak to harvest."),
        dict(t0=t_breath - 1.6, text="Rising to gulp air. Electric eels breathe air at the surface."),
        dict(t0=t_breath + 1.2, text="Cruising and sensing with weak ~10 V pulses."),
        dict(t0=events[0]["t0"] - 0.3, text=events[0]["story"]),
        dict(t0=events[0]["t0"] + 3.0, text="Cruising toward the hide tube."),
        dict(t0=events[1]["t0"] - 0.3, text=events[1]["story"]),
        dict(t0=t_drop, text="A keeper drops food through the hatch. The eel senses it sinking."),
        dict(t0=events[2]["t0"] - 0.3, text=events[2]["story"]),
        dict(t0=events[2]["t0"] + events[2]["vis"] + 1.0,
             text="Energy reaches storage; the beacon warns keepers the water is live."),
        dict(t0=t_refresh, text="Enough surplus: the e-paper display refreshes with the new volley."),
        dict(t0=t_refresh + 3.0, text="Back to cruising. One eel collects ~0.1 J on a typical day."),
    ]
    captions.sort(key=lambda c: c["t0"])
    for i, c in enumerate(captions):
        c["t1"] = captions[i + 1]["t0"] if i + 1 < len(captions) else period
        c["t0"], c["t1"] = round(c["t0"], 3), round(c["t1"], 3)

    log = []
    for ev in events:
        log.append(dict(t=round(ev["t0"], 3), text=f"{ev['label']}: {plate_names[ev['plus']]}+ / "
                                                     f"{plate_names[ev['minus']]}−, ΔV {ev['dV']:.1f} V, "
                                                     f"{ev['mJ_event']:.1f} mJ collected"))
    log.append(dict(t=round(t_breath, 3), text="Surfaced to breathe"))
    log.append(dict(t=round(t_drop, 3), text="Food dropped through the hatch"))
    log.append(dict(t=round(t_refresh, 3), text="Display refreshed from surplus energy"))
    log.sort(key=lambda x: x["t"])

    clean = []
    for ev in events:
        e = {k: v for k, v in ev.items() if k not in ("story",)}
        e["flashes"] = [[round(a, 3), round(b, 3)] for a, b in ev["flashes"]]
        for k in ("t0", "t_mid", "vis"):
            e[k] = round(float(e[k]), 3)
        clean.append(e)
    markers = [dict(t=round(e["t0"], 3), kind=e["kind"], label=e["label"]) for e in clean]
    markers += [dict(t=round(t_breath, 3), kind="breath", label="Breath"),
                dict(t=round(t_drop, 3), kind="food", label="Food dropped"),
                dict(t=round(t_refresh, 3), kind="display", label="Display refresh")]
    return dict(
        period=round(period, 4), fps=FPS, rest_t=round(rest_t, 3), plates=plate_names,
        constants=dict(sigma_S_per_m=SIGMA, current_A=I_EEL, organ_V=V_ORGAN, water_W=P_WATER, pair_ohm=R_PAIR,
                       pulse_s=T_PULSE, converter_eff=ETA, diode_V=DIODE, lv_V=LV_VOLTS, lv_plate_dV=round(lv_dV, 3),
                       supercap_F=SC_C, supercap_start_J=SC_E0, catch_uF=47, phone_J=PHONE_J,
                       typical_day_J=TYPICAL_DAY_J, packet_travel_s=PACKET_TRAVEL),
        events=clean, captions=captions, log=log, markers=sorted(markers, key=lambda m: m["t"]), series=series)


# --------------------------------------------------------------------------
# Part B - artificial electric organ (eel-inspired hydrogel stack)
# --------------------------------------------------------------------------

ORGAN_PERIOD = 4.0


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
    ex, centers = {}, []
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
    zf = EZ + ER * 0.95
    channels = []
    ion_nodes = []
    for kind, count, a, b, mesh_r, matname in (("Na+", 4, hs_x - 0.003, centers[4] + 0.002, 0.0025, "ion_cation"),
                                               ("Cl-", 3, hs_x + 0.003, centers[0] - 0.002, 0.0032, "ion_anion")):
        mesh = sphere(mesh_r, 12, 8)
        for k in range(count):
            dy = (-0.006, 0.004, -0.001, 0.007)[k]
            ph = k / count
            line = Polyline([(a, EYc + dy, zf), (b, EYc + dy, zf)])
            keys = packet_like_keys(line, ph)
            n = Node(f"{kind} ion {k + 1}", mesh, matname,
                     "Positive sodium ion leaving the high-salt gel through the cation membrane." if kind == "Na+" else
                     "Negative chloride ion leaving the high-salt gel through the anion membrane.",
                     t=tuple(keys[0][1]))
            channels.append(Channel(n, "translation", [t for t, _ in keys], [p for _, p in keys]))
            ion_nodes.append(n)
    ions = Node("Ion flow", children=[
        Node("Na+ arrow", arrow((hs_x, ay, EZ), (centers[4] + 0.004, ay, EZ), 0.0016, 0.0042, 0.008), "ion_cation"),
        Node("Cl- arrow", arrow((hs_x, ay, EZ), (centers[0] - 0.004, ay, EZ), 0.0016, 0.0042, 0.008), "ion_anion"),
        Node("Current arrow", arrow((-total / 2, ay + 0.016, EZ), (total / 2, ay + 0.016, EZ), 0.0018, 0.0048, 0.009),
             "current_arrow", "Conventional current runs toward +x, so the +x end is the positive terminal."),
    ] + ion_nodes)
    # Lay the two exhibits side by side so the arrows never sit in front of the stack.
    for n in (cradle, stack, sleeve, caps, elec, leads, implant):
        shift(n, T(-0.085, 0, -0.01))
    for n in (exploded, ions):
        shift(n, T(0.10, 0, -0.075))
    for ch in channels:
        ch.values = ch.values + np.array([0.10, 0, -0.075])
        ch.node.t = tuple(ch.values[0])
    base = Node("Base plate", box((0.38, 0.008, 0.20), (-0.005, 0.004, 0.0)), "base_plate")
    return [base, cradle, stack, sleeve, caps, elec, leads, implant, exploded, ions], channels, {"cells": UNITS}


def packet_like_keys(line, phase):
    """Looping LINEAR keys: move along the line once per ORGAN_PERIOD, then jump back."""
    w = (1 - phase) * ORGAN_PERIOD          # time the ion reaches the end and wraps
    keys = [(0.0, line.at(phase * line.length)), (w - 1e-3, line.at(line.length))]
    if w < ORGAN_PERIOD - 1e-6:
        keys += [(w, line.at(0.0)), (ORGAN_PERIOD, line.at(phase * line.length))]
    else:
        keys.append((ORGAN_PERIOD, line.at(0.0)))
    return keys


def shift(node, M):
    if node.t is not None:          # animated nodes move by their translation, not their mesh
        node.t = tuple((M @ np.array([*node.t, 1.0]))[:3])
    else:
        node.prims = [(m.xform(M), mt) for m, mt in node.prims]
    for c in node.children:
        shift(c, M)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def static_leaves(roots):
    """World-space (path, node, prims) for the frozen OBJ / STL exports."""
    for r in roots:
        for path, node, W in r.walk():
            if node.anim_only or not node.prims:
                continue
            if node.static is not None:
                yield path, node, node.static
            elif node.skin is None:
                yield path, node, [(m.xform(W), mt) for m, mt in node.prims]


def write_glb(path, roots, title, channels=(), clip=None):
    blob = bytearray()
    views, accessors, meshes, nodes, skins = [], [], [], [], []
    node_index, mesh_cache, skin_cache, skin_refs = {}, {}, {}, []
    used = []

    def view(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        blob.extend(data)
        return len(views) - 1

    def accessor(arr, typ, comp, target=None, minmax=False):
        a = {"bufferView": view(arr.tobytes(), target), "componentType": comp, "count": int(len(arr)), "type": typ}
        if minmax:
            flat_ = arr.reshape(len(arr), -1)
            a["min"] = [float(v) for v in flat_.min(axis=0)]
            a["max"] = [float(v) for v in flat_.max(axis=0)]
        accessors.append(a)
        return len(accessors) - 1

    def mesh_index(node):
        key = tuple((id(m), mt) for m, mt in node.prims)
        if key in mesh_cache:
            return mesh_cache[key]
        prims = []
        for m, mt in node.prims:
            if mt not in used:
                used.append(mt)
            pos = m.pos.astype(np.float32)
            big = len(pos) > 65535
            attrs = {"POSITION": accessor(pos, "VEC3", 5126, 34962, True),
                     "NORMAL": accessor(m.nrm.astype(np.float32), "VEC3", 5126, 34962)}
            if m.joints is not None:
                attrs["JOINTS_0"] = accessor(m.joints.astype(np.uint16), "VEC4", 5123, 34962)
                attrs["WEIGHTS_0"] = accessor(m.weights.astype(np.float32), "VEC4", 5126, 34962)
            idx = m.idx.astype(np.uint32 if big else np.uint16).ravel()
            prims.append({"attributes": attrs, "indices": accessor(idx, "SCALAR", 5125 if big else 5123, 34963),
                          "material": mt})
        meshes.append({"name": node.name, "primitives": prims})
        mesh_cache[key] = len(meshes) - 1
        return mesh_cache[key]

    def add(node):
        i = len(nodes)
        entry = {"name": node.name}
        nodes.append(entry)
        node_index[id(node)] = i
        if node.desc:
            entry["extras"] = {"description": node.desc}
        if node.t is not None:
            entry["translation"] = [float(v) for v in node.t]
        if node.r is not None:
            entry["rotation"] = [float(v) for v in node.r]
        if node.s is not None:
            entry["scale"] = [float(v) for v in node.s]
        if node.prims:
            entry["mesh"] = mesh_index(node)
        if node.skin is not None:
            skin_refs.append((entry, node.skin))
        if node.children:
            entry["children"] = [add(c) for c in node.children]
        return i

    root_ids = [add(r) for r in roots]
    for entry, skin in skin_refs:
        if id(skin) not in skin_cache:
            ibm = np.array([m.T.ravel() for m in skin.ibm], dtype=np.float32)   # column-major
            skins.append({"joints": [node_index[id(j)] for j in skin.joints],
                          "inverseBindMatrices": accessor(ibm, "MAT4", 5126)})
            skin_cache[id(skin)] = len(skins) - 1
        entry["skin"] = skin_cache[id(skin)]
    for m in meshes:
        for p in m["primitives"]:
            p["material"] = used.index(p["material"])

    animations = []
    if channels:
        samplers, chans, time_cache = [], [], {}
        for ch in channels:
            times = ch.times.astype(np.float32)
            key = times.tobytes()
            if key not in time_cache:
                time_cache[key] = accessor(times, "SCALAR", 5126, minmax=True)
            vals = ch.values.astype(np.float32)
            out = accessor(vals, "VEC4" if ch.path == "rotation" else "VEC3", 5126)
            samplers.append({"input": time_cache[key], "output": out, "interpolation": ch.interp})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": node_index[id(ch.node)], "path": ch.path}})
        animations.append({"name": clip or title, "samplers": samplers, "channels": chans})

    mats, ext_used = [], set()
    for name in used:
        m = MATERIALS[name]
        rgb = [float(v) for v in srgb_to_linear(m["color"])]
        e = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": rgb + [m["alpha"]],
                                                    "metallicFactor": m["metal"], "roughnessFactor": m["rough"]}}
        if any(m["emit"]):
            e["emissiveFactor"] = [float(v) for v in srgb_to_linear(m["emit"])]
            if m["glow"] > 1:
                e["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": m["glow"]}}
                ext_used.add("KHR_materials_emissive_strength")
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
    if skins:
        gltf["skins"] = skins
    if animations:
        gltf["animations"] = animations
    if ext_used:
        gltf["extensionsUsed"] = sorted(ext_used)
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
    out = [f"# {title}", "# Units: metres, +Y up. A single frozen moment of the animation.",
           "# Generated by tools/build_models.py", f"mtllib {base}.mtl"]
    used, off = [], 1
    for p, node, prims in static_leaves(roots):
        for m, mt in prims:
            if mt not in used:
                used.append(mt)
            name = p.replace("/", ".") + (f".{mt}" if len(prims) > 1 else "")
            out.append(f"o {_safe(name)}")
            out.append("\n".join("v %.5f %.5f %.5f" % tuple(v) for v in m.pos))
            out.append("\n".join("vn %.4f %.4f %.4f" % tuple(v) for v in m.nrm))
            out.append(f"usemtl {mt}")
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
    tris = [m.pos[m.idx] for _, node, prims in static_leaves(roots) if node.stl for m, _ in prims]
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


def export(stem, roots, title, channels=(), clip=None):
    base = os.path.join(OUT, stem)
    write_glb(base + ".glb", roots, title, channels, clip)
    write_obj(base + ".obj", roots, title)
    n_stl = write_stl(base + ".stl", roots, title)
    leaves = [(n, m) for r in roots for _, n, _ in r.walk() for m, _ in n.prims]
    verts = sum(len(m.pos) for _, m in leaves)
    tris = sum(len(m.idx) for _, m in leaves)
    sizes = {ext: os.path.getsize(base + ext) for ext in (".glb", ".obj", ".mtl", ".stl")}
    print(f"{stem}: {len(leaves)} primitives, {verts:,} vertices, {tris:,} triangles (STL {n_stl:,}), "
          f"{len(channels)} animation channels")
    print("   " + ", ".join(f"{e} {s / 1e6:.2f} MB" for e, s in sizes.items()))


def main():
    os.makedirs(OUT, exist_ok=True)
    roots, channels, timeline, stats = scene_habitat()
    export("eel-habitat-harvester", roots, "Passive eel habitat harvester", channels, "Swim and harvest (loop)")
    with open(os.path.join(OUT, "eel-habitat-harvester.timeline.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(timeline, f, separators=(",", ":"))
    print("   ", stats)
    roots, channels, stats = scene_organ()
    export("artificial-electric-organ", roots, "Artificial electric organ (eel-inspired hydrogel stack)", channels,
           "Ion flow (loop)")
    print("   ", stats)


if __name__ == "__main__":
    main()

"""The viewer: a window onto the practice, for someone watching from outside.

Started by the engine's `viewer` operation, it serves a page (three.js, in `viewer_static/`) on 127.0.0.1 that shows what the
body *does*: the bones moving, the muscles working along their paths, where the weight falls in each foot, the breath and the
heart in numbers. The felt sense, the practitioner's own words about the body, is private and never served here.

Python's standard library only: a ThreadingHTTPServer in a daemon thread, Server-Sent Events for the live stream. The port is
WUJI_VIEWER_PORT, or 7341, or the next free one after it.

    GET /             the page (viewer_static/index.html), and its assets
    GET /model.bin    the visual model, built once (layout below); gzip when the browser accepts it
    GET /stream       Server-Sent Events: one JSON frame per message, about 25 a second while someone watches

Threads. The engine loop calls `publish(engine)` once per frame (25 Hz). Only while a client is connected does it build a frame:
the pose's kinematics and tendon paths on a MjData of the viewer's own (about 0.3 ms), and a JSON string it hands over under
a condition variable. Each SSE connection has its own thread that waits for the next frame and writes it; a slow or vanished
browser blocks only that thread, never the engine. A client that falls behind skips frames and always gets the latest.

model.bin
    bytes 0..3   b"WUJI"
    bytes 4..7   uint32 little-endian: length H of the JSON header
    bytes 8..    the JSON header (UTF-8), padded with spaces so the blobs start on a 4-byte boundary
    then         the blobs, little-endian, each at header["blobs"][name] = {"offset" (from the start of the blobs), "dtype",
                 "shape"}:
                   vert   float32 (nvert, 3)  every mesh's vertices, in its mesh frame, concatenated
                   face   uint16  (nface, 3)  every mesh's triangles, indices local to their mesh, concatenated
    header       {"version": 1,
                  "nbody": 102, "body_names": [...], "root_body": 1,
                  "meshes": [{"vert": first vertex, "nvert", "face": first face, "nface"}, ...],      (per MuJoCo mesh)
                  "geoms": [{"body", "mesh", "pos": [x, y, z], "quat": [w, x, y, z]}, ...],         (the 102 bones)
                  "tendons": [name, ...],                                                            (424)
                  "wrap_capacity": the most path points a frame can carry (two per wrap object),
                  "muscles": [{"name", "tendon", "region", "side", "f0"}, ...],                      (416)
                  "regions": [...], "frc_l": 2.8, "ic_l": 3.0, "blobs": {...}}
    The bones are the mesh geoms of group 0; groups 3 and 4 (collision and wrapping shapes) are left out. A geom's pos and
    quat place its mesh in its body's frame (MuJoCo has already folded any mesh_pos / mesh_quat offset into them).
    MuJoCo's world is z-up; the body faces -y, its right side is -x.

Stream frame (`data:` of each SSE message, JSON). Arrays too big for JSON numbers are base64 of little-endian binary:
    {"t", "stance", "moving",
     "bodies":  b64 float32 (nbody, 7): xpos, then xquat (w, x, y, z), of every body,
     "wrapnum": b64 uint16 (ntendon): how many path points each tendon has this frame,
     "wrap":    b64 float32 (sum wrapnum, 3): the points, tendon after tendon, world frame; NaN marks a pulley (a break),
     "act":     b64 float32 (nu): each muscle's activation 0..1,
     "feet":    {"left": {"cop": [x, y, z], "fz": N, "share", "loaded"}, "right": ...},
     "weight":  {"left", "right"},
     "effort":  {"total": mean squared activation, "regions": {region: 0..1}},
     "tension": {region: held level 0..1},
     "breath":  breath.state() (numbers only: phase, lung_volume_l, rate_bpm, tidal_l, belly_cm, chest_cm, nostrils, co2_mmhg,
                heart_bpm, settled, ...), or null without a breath model}
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .anatomy import REGION_ORDER
from .senses import region_effort

if TYPE_CHECKING:
    from .engine import Engine

DEFAULT_PORT = 7341
PORTS_TO_TRY = 20
STATIC = Path(__file__).parent / "viewer_static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
MIN_INTERVAL = 1 / 30  # never build frames faster than this, whatever the engine's pace
BONE_GROUP = 0


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def build_model_bin(m: mujoco.MjModel, anatomy) -> bytes:
    """The visual model (layout in the module docstring)."""
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""
    bones = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == BONE_GROUP]
    used = sorted({int(m.geom_dataid[g]) for g in bones})
    meshes, verts, faces, index = [], [], [], {}
    nv = nf = 0
    for mid in used:
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        if vn > 65535:
            raise ValueError(f"mesh {mid} has {vn} vertices, too many for uint16 indices")
        index[mid] = len(meshes)
        meshes.append({"name": name(mujoco.mjtObj.mjOBJ_MESH, mid), "vert": nv, "nvert": vn, "face": nf, "nface": fn})
        verts.append(m.mesh_vert[va:va + vn])
        faces.append(m.mesh_face[fa:fa + fn])
        nv, nf = nv + vn, nf + fn
    vert = np.concatenate(verts).astype("<f4")
    face = np.concatenate(faces).astype("<u2")
    geoms = [{"body": int(m.geom_bodyid[g]), "mesh": index[int(m.geom_dataid[g])],
              "pos": [round(float(x), 6) for x in m.geom_pos[g]], "quat": [round(float(x), 7) for x in m.geom_quat[g]]}
             for g in bones]
    tendons = [name(mujoco.mjtObj.mjOBJ_TENDON, t) for t in range(m.ntendon)]
    muscles = [{"name": anatomy.muscles[i], "tendon": int(m.actuator_trnid[i, 0]) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON else -1,
                "region": str(anatomy.muscle_region[i]), "side": str(anatomy.muscle_side[i]), "f0": round(float(anatomy.f0[i]), 1)}
               for i in range(m.nu)]
    try:  # the breath's constants, for scaling the lungs on the page (read only)
        from .breath import FRC_L, IC_L  # noqa: PLC0415
    except Exception:
        FRC_L, IC_L = 2.8, 3.0
    blobs = {"vert": {"offset": 0, "dtype": "float32", "shape": list(vert.shape)},
             "face": {"offset": vert.nbytes, "dtype": "uint16", "shape": list(face.shape)}}
    header = {"version": 1, "nbody": int(m.nbody), "body_names": [name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)],
              "root_body": 1, "meshes": meshes, "geoms": geoms, "tendons": tendons,
              "wrap_capacity": 2 * int(m.nwrap), "muscles": muscles,
              "regions": list(REGION_ORDER), "frc_l": FRC_L, "ic_l": IC_L, "blobs": blobs}
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * (-(8 + len(h)) % 4)
    return b"WUJI" + struct.pack("<I", len(h)) + h + vert.tobytes() + face.tobytes()


class Viewer:
    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        m = engine.body.m
        self.m = m
        self.d = mujoco.MjData(m)  # the viewer's own: never the solvers'
        a = engine.body.a
        self.region_masks = {r: a.muscle_region == r for r in REGION_ORDER}
        self.f0 = a.f0
        self.server: ThreadingHTTPServer | None = None
        self.url: str | None = None
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.seq = 0
        self.clients = 0
        self.last_build = 0.0
        self.build_ms = 0.0  # what the last frame cost the engine loop
        self._model: dict[str, bytes] = {}
        self._model_lock = threading.Lock()

    # ---------- the server ----------

    def start(self) -> str:
        if self.url is not None:
            return self.url
        try:
            first = int(os.environ.get("WUJI_VIEWER_PORT", DEFAULT_PORT))
        except ValueError:
            first = DEFAULT_PORT
        handler = type("Handler", (_Handler,), {"viewer": self})
        error: OSError | None = None
        for port in range(first, first + PORTS_TO_TRY):
            try:
                self.server = _Server(("127.0.0.1", port), handler)
                break
            except OSError as e:
                error = e
        else:
            raise RuntimeError(f"no free port for the viewer from {first} to {first + PORTS_TO_TRY - 1} ({error})")
        port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, name="wuji-viewer", daemon=True).start()
        self.url = f"http://127.0.0.1:{port}/"
        print(f"[wuji] viewer at {self.url}", file=sys.stderr)
        return self.url

    def model_bin(self, gz: bool) -> bytes:
        with self._model_lock:
            if not self._model:
                raw = build_model_bin(self.m, self.engine.body.a)
                self._model = {"raw": raw, "gzip": gzip.compress(raw, compresslevel=5)}
            return self._model["gzip" if gz else "raw"]

    # ---------- frames (engine thread) ----------

    def publish(self, engine: "Engine") -> None:
        """Called by the engine loop every frame; builds a frame only while someone is watching."""
        if self.clients <= 0:
            return
        now = time.monotonic()
        if now - self.last_build < MIN_INTERVAL:
            return
        self.last_build = now
        t0 = time.perf_counter()
        frame = self.build(engine)
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()
        self.build_ms = 1000 * (time.perf_counter() - t0)

    def build(self, engine: "Engine") -> bytes:
        b, m, d = engine.body, self.m, self.d
        d.qpos[:] = b.q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_tendon(m, d)  # needs only the kinematics: wrap paths, lengths

        pts = d.wrap_xpos.reshape(-1, 3)
        obj = d.wrap_obj.reshape(-1)
        num = d.ten_wrapnum.astype(np.int64)
        idx = np.concatenate([np.arange(a, a + n) for a, n in zip(d.ten_wrapadr, num)]) if num.sum() else np.zeros(0, int)
        wrap = pts[idx].astype(np.float32)
        wrap[obj[idx] == -2] = np.nan  # a pulley: the path branches here

        s = b.solution
        feet, weight, act, total, regions = {}, {"left": 0.5, "right": 0.5}, np.zeros(m.nu), 0.0, {}
        if s is not None:
            act = s.act
            total = float(s.effort)
            fz = {k: max(0.0, float(s.grf[k]["force"][2])) if k in s.grf else 0.0 for k in ("left", "right")}
            sum_fz = sum(fz.values()) or 1.0
            for side in ("left", "right"):
                share = fz[side] / sum_fz
                weight[side] = round(share, 4)
                cop = s.grf[side]["cop"] if side in s.grf and fz[side] > 1.0 else None
                feet[side] = {"cop": None if cop is None else [round(float(x), 4) for x in cop], "fz": round(fz[side], 1),
                              "share": round(share, 4), "loaded": bool(b.loaded.get(side, True))}
            regions = {r: round(region_effort(act, self.f0, sel), 4) for r, sel in self.region_masks.items()}

        breath = engine.breath.state() if engine.breath is not None else None
        frame = {
            "t": round(b.t, 3),
            "stance": b.posture.stance,
            "moving": b.moving,
            "bodies": _b64(np.hstack([d.xpos, d.xquat]), "<f4"),
            "wrapnum": _b64(num, "<u2"),
            "wrap": _b64(wrap, "<f4"),
            "act": _b64(act, "<f4"),
            "feet": feet,
            "weight": weight,
            "effort": {"total": round(total, 6), "regions": regions},
            "tension": {k: round(float(v), 3) for k, v in b.tension.items()},
            "breath": breath,
        }
        return json.dumps(frame, separators=(",", ":"), default=_plain).encode("utf-8")

    # ---------- frames (SSE threads) ----------

    def watch(self, write) -> None:
        """Streams frames to one client until it goes away."""
        with self.cond:
            self.clients += 1
        try:
            seen = self.seq  # wait for a fresh frame: the one kept may be stale (built before anyone watched)
            while True:
                with self.cond:
                    fresh = self.cond.wait_for(lambda: self.seq != seen, timeout=2.0)
                    frame, seq = self.frame, self.seq
                if fresh and frame is not None:
                    seen = seq
                    write(b"data: " + frame + b"\n\n")
                else:
                    write(b": the body is still\n\n")  # keeps the connection, and finds a closed one
        finally:
            with self.cond:
                self.clients -= 1

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


def _plain(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(f"not JSON: {type(x)}")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second socket bind a port already in use, which would hide a busy port from the
    # fallback; elsewhere it only skips TIME_WAIT. Exclusive use on Windows, reuse elsewhere.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _Handler(BaseHTTPRequestHandler):
    viewer: Viewer
    timeout = 30  # a client that stops reading for this long is let go
    server_version = "wuji-viewer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:  # stdout is the protocol; stay quiet on stderr too
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path == "/stream":
                return self._stream()
            if path == "/model.bin":
                gz = "gzip" in self.headers.get("Accept-Encoding", "")
                body = self.viewer.model_bin(gz)
                return self._send(200, body, "application/octet-stream", {"Content-Encoding": "gzip"} if gz else {},
                                  cache="no-cache")
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            file = (STATIC / name).resolve()
            if STATIC.resolve() not in file.parents or not file.is_file() or file.suffix not in TYPES:
                return self._send(404, b"not here\n", "text/plain; charset=utf-8")
            return self._send(200, file.read_bytes(), TYPES[file.suffix], cache="no-cache")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            self.close_connection = True

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b"retry: 2000\n\n")
        self.wfile.flush()

        def write(chunk: bytes) -> None:
            self.wfile.write(chunk)
            self.wfile.flush()

        self.viewer.watch(write)

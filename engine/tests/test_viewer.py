"""The viewer: the `viewer` operation starts it, and it serves the page, the visual model and a live stream of frames."""
import base64
import json
import os
import struct
import subprocess
import sys
import threading
import time
import queue
import urllib.request

import numpy as np
import pytest


class Engine:
    def __init__(self, port: int):
        env = {**os.environ, "WUJI_VIEWER_PORT": str(port)}
        self.p = subprocess.Popen([sys.executable, "-m", "wuji.engine"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, env=env)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.next_id = 0

    def _read(self):
        for line in self.p.stdout:
            self.lines.put(json.loads(line))

    def call(self, op, timeout=60, **args):
        self.next_id += 1
        self.p.stdin.write(json.dumps({"id": self.next_id, "op": op, "args": args}) + "\n")
        self.p.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            msg = self.lines.get(timeout=max(0.1, deadline - time.monotonic()))
            if msg.get("id") == self.next_id:
                return msg

    def close(self):
        try:
            self.call("shutdown")
        finally:
            self.p.wait(timeout=10)


@pytest.fixture(scope="module")
def viewer():
    e = Engine(port=7391)  # away from the default, so a viewer someone is watching is left alone
    r = e.call("viewer")
    assert r["ok"], r
    yield e, r["result"]
    e.close()


def get(url, headers=None, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout) as resp:
        return resp.status, resp.headers, resp.read()


def test_the_op_returns_a_url_and_is_idempotent(viewer):
    engine, result = viewer
    assert result["url"].startswith("http://127.0.0.1:") and result["url"].endswith("/")
    assert "felt sense" in result["text"]
    again = engine.call("viewer")
    assert again["ok"] and again["result"]["url"] == result["url"]


def test_the_page_and_its_assets(viewer):
    _, result = viewer
    status, headers, body = get(result["url"])
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"importmap" in body and b"three@0.160.0" in body
    for asset in ("viewer.js", "viewer.css"):
        assert get(result["url"] + asset)[0] == 200
    with pytest.raises(urllib.error.HTTPError) as e:
        get(result["url"] + "../engine.py")
    assert e.value.code == 404


def test_model_bin_holds_the_bones_and_the_muscles(viewer):
    _, result = viewer
    status, headers, raw = get(result["url"] + "model.bin")
    assert status == 200 and raw[:4] == b"WUJI"
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen])
    base = 8 + hlen
    assert base % 4 == 0
    vb, fb = header["blobs"]["vert"], header["blobs"]["face"]
    vert = np.frombuffer(raw, "<f4", count=vb["shape"][0] * 3, offset=base + vb["offset"]).reshape(-1, 3)
    face = np.frombuffer(raw, "<u2", count=fb["shape"][0] * 3, offset=base + fb["offset"]).reshape(-1, 3)
    assert len(raw) == base + fb["offset"] + face.nbytes
    assert len(header["geoms"]) == 102  # the bones; collision and wrapping shapes are left out
    assert len(header["meshes"]) == 102  # the meshes the bones use (one of the model's 103 is unused)
    assert len(vert) == sum(m["nvert"] for m in header["meshes"]) and len(face) == sum(m["nface"] for m in header["meshes"])
    assert len(vert) > 140_000 and len(face) > 280_000
    for m in header["meshes"]:
        assert face[m["face"]:m["face"] + m["nface"]].max() < m["nvert"]
    assert len(header["tendons"]) == 424 and len(header["muscles"]) == 416
    assert all(0 <= mu["tendon"] < 424 for mu in header["muscles"])
    assert {mu["region"] for mu in header["muscles"]} == set(header["regions"])
    assert {mu["side"] for mu in header["muscles"]} == {"left", "right"}
    # gzip when asked for it, same content
    _, gz_headers, gz = get(result["url"] + "model.bin", {"Accept-Encoding": "gzip"})
    assert gz_headers["Content-Encoding"] == "gzip" and len(gz) < len(raw)


def test_stream_yields_frames(viewer):
    _, result = viewer
    with urllib.request.urlopen(result["url"] + "stream", timeout=20) as resp:
        assert resp.headers["Content-Type"] == "text/event-stream"
        deadline = time.monotonic() + 10
        frame = None
        while time.monotonic() < deadline:
            line = resp.readline().decode("utf-8")
            if line.startswith("data: "):
                frame = json.loads(line[6:])
                break
    assert frame is not None, "no frame within 10 s"
    assert {"t", "stance", "bodies", "wrapnum", "wrap", "act", "feet", "weight", "effort", "tension", "breath"} <= set(frame)
    bodies = np.frombuffer(base64.b64decode(frame["bodies"]), "<f4").reshape(-1, 7)
    assert bodies.shape == (102, 7)
    wrapnum = np.frombuffer(base64.b64decode(frame["wrapnum"]), "<u2")
    wrap = np.frombuffer(base64.b64decode(frame["wrap"]), "<f4").reshape(-1, 3)
    assert len(wrapnum) == 424 and len(wrap) == wrapnum.sum() > 1000
    act = np.frombuffer(base64.b64decode(frame["act"]), "<f4")
    assert len(act) == 416 and act.min() >= 0 and act.max() <= 1
    assert abs(frame["weight"]["left"] + frame["weight"]["right"] - 1) < 0.01
    assert set(frame["effort"]["regions"]) == {"feet", "calves", "thighs", "hips", "lower_back", "belly", "shoulders", "arms", "hands"}
    head = bodies[:, 2].max()
    assert 1.4 < head < 1.9, "standing upright"
    breath = frame["breath"]
    if breath is not None:  # the breath model is built separately
        assert {"phase", "lung_volume_l", "rate_bpm", "belly_cm", "chest_cm", "nostrils", "co2_mmhg", "heart_bpm"} <= set(breath)
    assert "text" not in frame and "text" not in (breath or {}), "the felt sense stays private"

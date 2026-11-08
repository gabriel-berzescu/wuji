"""The engine process, spoken to as the server speaks to it: JSON Lines over stdio (docs/engine-protocol.md)."""
import json
import subprocess
import sys
import threading
import time
import queue

import pytest


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-m", "wuji.engine"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.next_id = 0
        self.events = []

    def _read(self):
        for line in self.p.stdout:
            self.lines.put(json.loads(line))

    def call(self, op, timeout=30, **args):
        self.next_id += 1
        self.p.stdin.write(json.dumps({"id": self.next_id, "op": op, "args": args}) + "\n")
        self.p.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            msg = self.lines.get(timeout=max(0.1, deadline - time.monotonic()))
            if "event" in msg:
                self.events.append(msg)
            elif msg.get("id") == self.next_id:
                return msg

    def close(self):
        try:
            self.call("shutdown")
        finally:
            self.p.wait(timeout=10)


@pytest.fixture(scope="module")
def engine():
    e = Engine()
    yield e
    e.close()


def test_hello(engine):
    r = engine.call("hello")
    assert r["ok"] and r["result"]["muscles"] == 416 and r["result"]["model"] == "myofullbody"


def test_sense_brief_and_full(engine):
    brief = engine.call("sense")["result"]
    assert brief["text"].startswith("Wuji stance")
    assert {"stance", "weight", "effort", "joints_deg"} <= set(brief["body"])
    full = engine.call("sense", detail="full")["result"]
    assert "Joints:" in full["text"] and len(full["text"]) > len(brief["text"])


def test_intend_tension_breathe_and_errors(engine):
    r = engine.call("intend", weight=0.6, sink=0.3, over=1.0)
    assert r["ok"] and r["result"]["accepted"] and r["result"]["arrives_in"] == 1.0
    assert engine.call("tension", region="belly", level=0.3)["result"]["accepted"]
    assert engine.call("breathe", pattern="abdominal", inhale=4, exhale=6)["ok"]
    bad = engine.call("intend", stance="crane")
    assert not bad["ok"] and "crane" in bad["error"]
    assert not engine.call("fly")["ok"]


def test_a_form_reports_its_steps(engine):
    r = engine.call("form", name="wuji")
    assert r["ok"] and r["result"]["steps"] == 4
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and not any(e["event"] == "form" and e["data"]["done"] for e in engine.events):
        engine.call("sense")
        time.sleep(0.5)
    steps = [e for e in engine.events if e["event"] == "form"]
    assert steps and steps[-1]["data"]["done"]


def test_reset(engine):
    r = engine.call("reset")
    assert r["ok"] and "wuji" in r["result"]["text"]

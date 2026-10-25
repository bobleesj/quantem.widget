"""A live monitor updates during work and releases its sampling thread."""

import threading
import time
from types import SimpleNamespace

import psutil
import pytest
import traitlets

from quantem.widget import ShowGPU, showgpu


def test_live_sampling_pause_refresh_resume_and_close(monkeypatch):
    samples = []
    received = threading.Event()

    def sample(**kwargs):
        samples.append(time.monotonic())
        if len(samples) >= 3:
            received.set()
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", sample)
    monitor = ShowGPU(interval=0.2)
    try:
        # The main execution thread is occupied; monitoring still advances.
        assert received.wait(timeout=2)
        monitor.running = False
        time.sleep(0.25)
        paused = monitor.snapshot["sequence"]
        time.sleep(0.25)
        assert monitor.snapshot["sequence"] == paused
        monitor.refresh()
        assert monitor.snapshot["sequence"] == paused + 1
        received.clear()
        monitor.running = True
        assert received.wait(timeout=2)
    finally:
        monitor.close()
    assert not monitor._worker.is_alive()
    after_close = len(samples)
    monitor.refresh()
    assert len(samples) == after_close


def test_failed_sample_is_visible_and_next_refresh_recovers(monkeypatch):
    def failed(**kwargs):
        raise OSError("Driver temporarily unavailable")

    monkeypatch.setattr(showgpu, "sample", failed)
    monitor = ShowGPU(running=False)
    try:
        assert "temporarily unavailable" in monitor.snapshot["error"]
        assert monitor.snapshot["gpus"] == []
        monkeypatch.setattr(
            showgpu,
            "sample",
            lambda **kwargs: {"gpus": [], "timestamp": time.time(), "error": ""},
        )
        monitor.refresh()
        assert monitor.snapshot["error"] == ""
    finally:
        monitor.close()


def test_refresh_requests_never_overlap_queries(monkeypatch):
    monitor = ShowGPU(running=False)
    entered = threading.Event()
    finish = threading.Event()
    calls = []

    def slow_sample(**kwargs):
        calls.append(1)
        entered.set()
        finish.wait(timeout=2)
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", slow_sample)
    worker = threading.Thread(target=monitor.refresh)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        monitor.refresh()
        assert len(calls) == 1
    finally:
        finish.set()
        worker.join(timeout=2)
        monitor.close()


def test_display_options_change_without_recreating_a_paused_monitor(monkeypatch):
    sampled_users = []

    def sample(*, users):
        sampled_users.append(users)
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", sample)
    monitor = ShowGPU(users=False, size="large", running=False)
    try:
        assert monitor.size == "large"
        assert sampled_users == [False]
        for size in ("small", "medium", "large", "xlarge"):
            monitor.size = size
            assert monitor.size == size
        # Resizing is a display change, not another driver query.
        assert sampled_users == [False]
        monitor.users = True
        assert sampled_users == [False, True]
        with pytest.raises(traitlets.TraitError, match="small"):
            monitor.size = "huge"
    finally:
        monitor.close()


def test_compute_users_are_grouped_per_gpu_and_missing_values_stay_missing(monkeypatch):
    telemetry = pytest.importorskip("quantem.gpu.device.telemetry")
    queried_users = []

    def status(*, processes):
        queried_users.append(processes)
        result = {"gpus": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}], "error": ""}
        if processes:
            result["processes"] = [
                {"gpu_uuid": "GPU-a", "pid": 1, "used_bytes": 100},
                {"gpu_uuid": "GPU-a", "pid": 2, "used_bytes": 200},
                {"gpu_uuid": "GPU-b", "pid": 1, "used_bytes": None},
                {"gpu_uuid": "GPU-a", "pid": 3, "used_bytes": 300},
            ]
        return result

    def process(pid=None):
        if pid == 3:
            raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(
            username=lambda: "example-user", memory_info=lambda: SimpleNamespace(rss=50)
        )

    monkeypatch.setattr(telemetry, "status", status)
    monkeypatch.setattr(showgpu.psutil, "Process", process)
    snapshot = showgpu.sample(users=True)
    assert snapshot["gpus"][0]["users"] == [
        {"name": "example-user", "used_bytes": 300, "processes": 2},
        {"name": None, "used_bytes": 300, "processes": 1},
    ]
    assert snapshot["gpus"][1]["users"] == [
        {"name": "example-user", "used_bytes": None, "processes": 1},
    ]
    assert "processes" not in snapshot
    snapshot = showgpu.sample(users=False)
    assert queried_users == [True, False]
    assert all("users" not in gpu for gpu in snapshot["gpus"])

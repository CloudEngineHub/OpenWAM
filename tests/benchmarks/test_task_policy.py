"""Native config identity and lazy single-model ownership."""

from copy import deepcopy
from types import SimpleNamespace

import labtasker
import pytest

from benchmarks.utils import task_policy as policy
from benchmarks.utils.eval_manifest import load_cached_manifest, load_manifest, publish_manifest, seal_manifest


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "dataloader:\n  num_frames: 33\n  height: 384\n  width: 320\nrun_name: ${now:%Y%m%d}\n"
    )
    (tmp_path / "checkpoint_step_1.safetensors").write_bytes(b"weights")
    return tmp_path


def test_native_overrides_and_checkpoint_selection(checkpoint):
    baseline = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint)])
    other = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint), "--denoise-steps", "7"])
    same = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint), "inference.denoise_steps=7"])
    assert other["identity"] == same["identity"] != baseline["identity"]
    assert other["effective_config"]["inference"]["denoise_steps"] == 7
    assert baseline["effective_config"]["run_name"] == "${now:%Y%m%d}"
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"changed")
    assert policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint)])["identity"] == baseline["identity"]
    (checkpoint / "checkpoint_step_2.safetensors").write_bytes(b"new weights")
    latest = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint)])
    assert latest["checkpoint"].endswith("checkpoint_step_2.safetensors")
    assert latest["identity"] != baseline["identity"]
    assert baseline["checkpoint"].endswith("checkpoint_step_1.safetensors")


def test_lazy_reload_only_on_model_change(checkpoint, monkeypatch):
    import websockets.sync.client

    first = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint)])
    second = policy.resolve_policy_model_spec(["--ckpt-dir", str(checkpoint), "--denoise-steps", "7"])
    args = SimpleNamespace(
        output_dir=checkpoint, server_python="python", host="127.0.0.1", port=9999, gpu=0, server_start_timeout=1
    )
    launched = []

    class Process:
        def __init__(self):
            self.code = None

        def poll(self):
            return self.code

    class Registry:
        def __init__(self):
            self.items = []

        def add(self, process, log):
            self.items.append((process, log))

        def terminate_all(self):
            for process, log in self.items:
                process.code = 0
                log.close()
            self.items.clear()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def send(self, value):
            pass

        def recv(self, **kw):
            return '{"type":"pong"}'

    def spawn(*args, **kw):
        assert all(p.poll() is not None for p in launched)
        process = Process()
        launched.append(process)
        return process

    monkeypatch.setattr(policy, "ProcessRegistry", Registry)
    monkeypatch.setattr(policy, "spawn_resource", spawn)
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **k: Connection())
    monkeypatch.setattr(labtasker, "cancellation_requested", lambda: False)
    server = policy.WorkerPolicyServer(args)
    try:
        server.load_or_reuse(first)
        (checkpoint / "checkpoint_step_1.safetensors").unlink()
        server.load_or_reuse(deepcopy(first))
        assert len(launched) == 1
        server.load_or_reuse(second)
        assert len(launched) == 2
        launched[-1].code = 1
        server.load_or_reuse(second)
        assert len(launched) == 3
    finally:
        server.close()
    assert all(p.poll() is not None for p in launched)


def test_cleanup_failure_is_fatal(monkeypatch):
    server = policy.WorkerPolicyServer(SimpleNamespace())

    def fail():
        raise OSError("cannot terminate")

    monkeypatch.setattr(server.owned, "terminate_all", fail)
    with pytest.raises(labtasker.FatalWorkerError):
        server.close()


def test_cache_rebuild_keeps_queued_manifest_immutable(tmp_path):
    cache_key = dict(benchmark="robotwin", task="test", mode="clean", seed=0)

    def build(seed):
        return seal_manifest(
            {"benchmark": "robotwin", "task": "test", "mode": "clean", "entries": [{"episode": 0, "seed": seed}]}
        )

    first, second = build(1), build(2)
    original = publish_manifest(tmp_path, cache_key, first)
    publish_manifest(tmp_path, cache_key, second)
    assert load_cached_manifest(tmp_path, cache_key)[1]["manifest_hash"] == second["manifest_hash"]
    assert load_manifest(original, expected_manifest_hash=first["manifest_hash"]) == first


def test_cache_error_preserves_original_reason(monkeypatch):
    from benchmarks.utils import eval_manifest

    def denied(*args):
        raise PermissionError("Permission denied: /shared/cache/index")

    monkeypatch.setattr(eval_manifest, "load_cached_manifest", denied)
    with pytest.raises(ValueError, match="Permission denied: /shared/cache/index"):
        eval_manifest.require_manifests("/shared/cache", [{"task": "example"}], "build-manifest")


@pytest.mark.parametrize("broken_hash", [None, "invalid", "sha256:../bad", "sha256:" + "z" * 64])
def test_corrupt_cache_index_suggests_rebuild(tmp_path, broken_hash):
    import json

    from benchmarks.utils import eval_manifest

    cache_key = dict(benchmark="robotwin", task="example", mode="demo_clean", seed=0)
    index = eval_manifest.manifest_index_path(tmp_path, cache_key)
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"manifest_hash": broken_hash}))
    with pytest.raises(ValueError) as error:
        eval_manifest.require_manifests(tmp_path, [cache_key], "python submit.py --operation build_manifest --rebuild")
    message = str(error.value)
    assert str(index) in message
    assert "invalid manifest_hash" in message
    assert "--operation build_manifest --rebuild" in message

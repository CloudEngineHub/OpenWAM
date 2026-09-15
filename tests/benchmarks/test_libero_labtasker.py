"""Manual two-operation submissions exercised against a real Labtasker server."""

import csv
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import labtasker
import numpy as np
import pytest

from benchmarks.utils.labtasker_utils import derive_task_id, list_submission_tasks, submit_tasks

ROOT = Path(__file__).resolve().parents[2]


def load_runtime(benchmark):
    path = ROOT / "benchmarks" / benchmark / "labtasker_runtime.py"
    spec = importlib.util.spec_from_file_location(f"{benchmark}_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("LABTASKER_URL", f"http://127.0.0.1:{free_port()}")
    monkeypatch.setenv("LABTASKER_QUEUE", "default")
    monkeypatch.delenv("LABTASKER_TOKEN", raising=False)
    monkeypatch.delenv("LABTASKER_SERVER_TOKEN", raising=False)
    port = os.environ["LABTASKER_URL"].rsplit(":", 1)[1]
    log = (tmp_path / "server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "labtasker_server", "serve", "--port", port, "--database", str(tmp_path / "db")],
        stdout=log,
        stderr=log,
    )
    try:
        with labtasker.Client() as client:
            for _ in range(100):
                try:
                    client.count_tasks()
                    break
                except labtasker.TransportError:
                    time.sleep(0.1)
            else:
                pytest.fail("server did not start")
            yield client
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


@pytest.mark.parametrize("benchmark", ["robotwin", "libero"])
def test_manual_stages_cache_retry_and_summary(benchmark, server, tmp_path, monkeypatch, capsys):
    rt = load_runtime(benchmark)
    parse = rt.parser if benchmark == "robotwin" else rt._build_parser
    if benchmark == "robotwin":
        monkeypatch.setattr(rt, "task_names", lambda: ["adjust_bottle"])
    extra = (
        ["--mode", "demo_clean", "--episodes", "3", "--episode-batch-size", "2"]
        if benchmark == "robotwin"
        else ["--smoke"]
    )
    options = [*extra, "--manifest-dir", str(tmp_path / "cache")]
    with pytest.raises(ValueError, match="Try building manifests"):
        rt.build_task_inputs(parse("submit").parse_args(options))
    assert server.count_tasks() == 0
    inputs = rt.build_task_inputs(parse("submit").parse_args(["--operation", "build_manifest", *options]))
    assert len(inputs) == 1 and "model" not in inputs[0]
    rt.submit(inputs, submission_id="manifest-1", route=f"openwam-{benchmark}-build_manifest")
    fake = tmp_path / "fake-python"
    source = "fake_robotwin_labtasker.py" if benchmark == "robotwin" else "fake_libero_labtasker_runtime.py"
    fake.write_text(f"#!{sys.executable}\n" + (Path(__file__).parent / source).read_text())
    fake.chmod(0o755)
    env_root = tmp_path / "environment"
    (env_root / "script").mkdir(parents=True)
    (env_root / "script/eval_policy.py").touch()
    out = tmp_path / "output"

    def worker(operation, route=None, **env):
        script = "labtasker_worker.py"
        cmd = [
            sys.executable,
            str(ROOT / "benchmarks" / benchmark / script),
            "--gpu",
            "0",
            "--idle-timeout",
            "0.2",
            "--output-dir",
            str(out),
            f"--{benchmark}-python",
            str(fake),
            f"--{benchmark}-path",
            str(env_root),
        ]
        if operation != "run_eval":
            cmd += ["--operation", operation]
        if route is not None:
            cmd += ["--route", route]
        if operation == "run_eval":
            cmd += ["--server-python", str(fake), "--port", str(free_port())]
        completed = subprocess.run(cmd, env=dict(os.environ, **env), capture_output=True, text=True, timeout=45)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    worker("build_manifest", FAKE_SERVER_FAIL="1", FAKE_SERVER_FAIL_GPU="0")
    tasks = list_submission_tasks(server, benchmark, "manifest-1")
    assert tasks[0].status == "succeeded", tasks[0].last_error
    expected_build_name = (
        "robotwin(demo_clean):adjust_bottle_build_manifest"
        if benchmark == "robotwin"
        else "libero(libero_spatial):task00_build_manifest"
    )
    assert tasks[0].name == expected_build_name
    assert not (out / "policy").exists()
    rt.submit(inputs, submission_id="manifest-2", route="custom-manifest")
    assert "Using cached manifest" in worker("build_manifest", route="custom-manifest")
    cache_index = next((tmp_path / "cache/index").glob("*.json"))
    cache_index.write_text("not json")
    rt.submit(inputs, submission_id="corrupt-cache", route=f"openwam-{benchmark}-build_manifest", max_attempts=1)
    worker("build_manifest")
    corrupt = list_submission_tasks(server, benchmark, "corrupt-cache")
    assert corrupt[0].status == "failed"
    assert "cannot read manifest cache index" in corrupt[0].last_error.message
    rt.submit(
        [item | {"rebuild": True} for item in inputs],
        submission_id="repair-cache",
        route=f"openwam-{benchmark}-build_manifest",
    )
    worker("build_manifest")
    rt.submit(
        [item | {"rebuild": True} for item in inputs],
        submission_id="rebuild",
        route=f"openwam-{benchmark}-build_manifest",
        max_attempts=2,
    )
    worker("build_manifest", FAKE_MANIFEST_FAIL="1")
    rebuilt = list_submission_tasks(server, benchmark, "rebuild")
    assert rebuilt[0].status == "failed" and rebuilt[0].attempt == 2
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("{}")
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"weights")
    args = rt.parse_submit_args([*options, "--", "--ckpt-dir", str(checkpoint)])
    batches = rt.build_task_inputs(args)
    assert batches and all(item["operation"] == "run_eval" for item in batches)
    rt.submit(batches, submission_id="eval-1", route=f"openwam-{benchmark}-run_eval")
    rt.submit(batches, submission_id="eval-1", route=f"openwam-{benchmark}-run_eval")
    worker("run_eval", FAKE_FAIL="1", FAKE_FAIL_ONCE="1", FAKE_FAIL_OPERATION="run_eval")
    completed = list_submission_tasks(server, benchmark, "eval-1")
    assert len(completed) == len(batches)
    assert all(task.status == "succeeded" for task in completed), [(t.status, t.last_error) for t in completed]
    assert sum(task.attempt == 2 for task in completed) == 1
    expected_names = (
        {"robotwin(demo_clean):adjust_bottle_unseen_01_02", "robotwin(demo_clean):adjust_bottle_unseen_02_02"}
        if benchmark == "robotwin"
        else {"libero(libero_spatial):task00_01_01"}
    )
    assert {task.name for task in completed} == expected_names
    assert all("model" not in task.result and "artifact_dir" in task.result for task in completed)
    monkeypatch.setattr(rt, "load_manifest", lambda *args, **kwargs: pytest.fail("summary read manifest cache"))
    rt.summarize("eval-1", out)
    summary_output = capsys.readouterr().out
    assert "Task summary\n" in summary_output
    assert ("Success/Episodes" if benchmark == "robotwin" else "Success/Trials") in summary_output
    assert "Success rate" in summary_output
    assert "Done/Total" in summary_output
    assert "Pending" in summary_output
    assert "Running" in summary_output
    assert "Failed" in summary_output
    assert "Output files\n" in summary_output
    assert ("Benchmark summary\n" if benchmark == "robotwin" else "Suite summary\n") in summary_output
    rows = list(csv.DictReader((out / "task_summary.csv").open()))
    assert len(rows) == 1
    assert rows[0]["success_rate"] == "1.0"
    if benchmark == "robotwin":
        benchmark_rows = list(csv.DictReader((out / "benchmark_summary.csv").open()))
        assert benchmark_rows[0]["episodes"] == rows[0]["episodes"] == "3"
        seen = rt.build_task_inputs(
            rt.parse_submit_args([*options, "--instruction-type", "seen", "--", "--ckpt-dir", str(checkpoint)])
        )
        assert seen[0]["manifest_hash"] == batches[0]["manifest_hash"]
        assert seen[0]["model"]["identity"] == batches[0]["model"]["identity"]
        rt.submit(batches + seen, submission_id="comparison", route=f"openwam-{benchmark}-run_eval")
        before = len(list((out / "policy").iterdir()))
        worker("run_eval")
        assert len(list((out / "policy").iterdir())) == before + 1
        rt.summarize("comparison", out)
        grouped = list(csv.DictReader((out / "benchmark_summary.csv").open()))
        assert {row["instruction_type"] for row in grouped} == {"seen", "unseen"}
        assert all(row["episodes"] == "3" and row["successes"] == "3" for row in grouped)


def test_interrupted_submission_is_completed_idempotently(server):
    inputs = [{"case": i} for i in range(3)]
    kwargs = dict(
        benchmark="test",
        submission_id="resume",
        route="test",
        names=["test"] * 3,
        max_attempts=3,
        priority=0,
    )

    class Interrupted:
        list_tasks = server.list_tasks

        def submit_task(self, args, **kw):
            if args["case"] == 1:
                raise OSError("network interrupted")
            return server.submit_task(args, **kw)

    with pytest.raises(OSError):
        submit_tasks(Interrupted(), inputs, **kwargs)
    ids = submit_tasks(server, inputs, **kwargs)
    assert len(set(ids)) == 3
    assert submit_tasks(server, inputs, **kwargs) == ids
    assert len(list_submission_tasks(server, "test", "resume")) == 3
    assert ids == [derive_task_id("resume", index) for index in range(3)]


def test_submission_supports_per_task_names(server):
    inputs = [{"case": 0}, {"case": 1}]
    submit_tasks(
        server,
        inputs,
        benchmark="test",
        submission_id="dynamic-names",
        route="test",
        names=["test:0", "test:1"],
        max_attempts=3,
        priority=0,
    )
    tasks = list_submission_tasks(server, "test", "dynamic-names")
    assert {task.name for task in tasks} == {"test:0", "test:1"}


@pytest.mark.parametrize("benchmark", ["robotwin", "libero"])
def test_benchmark_task_note_changes_display_name_only(benchmark, server):
    runtime = load_runtime(benchmark)
    item = (
        {"operation": "build_manifest", "mode": "demo_clean", "task": "adjust_bottle"}
        if benchmark == "robotwin"
        else {"operation": "build_manifest", "suite": "libero_spatial", "task_id": 0}
    )
    task_id = runtime.submit([item], submission_id=f"{benchmark}-note", route="note-only", note="candidate")[0]
    task = server.get_task(task_id)
    assert task.name.endswith("_candidate")
    assert "note" not in task.args


@pytest.mark.parametrize("benchmark", ["robotwin", "libero"])
def test_longer_manifest_serves_shorter_requested_prefix(benchmark, tmp_path, monkeypatch):
    from benchmarks.utils.eval_manifest import publish_manifest, seal_manifest
    from benchmarks.utils.rng_domain import encode_numpy_state

    runtime = load_runtime(benchmark)
    cache = tmp_path / "cache"
    if benchmark == "robotwin":
        entries = [
            {
                "episode": episode,
                "seed": 100000 + episode,
                "instructions": {"seen": f"seen-{episode}", "unseen": f"unseen-{episode}"},
            }
            for episode in range(100)
        ]
        manifest = seal_manifest(
            {"benchmark": "robotwin", "task": "adjust_bottle", "mode": "demo_clean", "entries": entries}
        )
        build_inputs = {
            "task": "adjust_bottle",
            "mode": "demo_clean",
        }
        argv = [
            "--mode",
            "demo_clean",
            "--tasks",
            "adjust_bottle",
            "--episodes",
            "20",
            "--episode-batch-size",
            "6",
            "--manifest-dir",
            str(cache),
        ]
    else:
        np_state = encode_numpy_state(np.random.RandomState(42).get_state())
        entries = [
            {"trial": trial, "init_state_index": trial, "pre_reset_numpy_state": np_state} for trial in range(50)
        ]
        manifest = seal_manifest(
            {
                "benchmark": "libero",
                "suite": "libero_spatial",
                "task_id": 0,
                "seed": 42,
                "init_state_asset": {"name": "a.pruned_init", "sha256": "sha256:" + "1" * 64},
                "bddl_asset": {"name": "a.bddl", "sha256": "sha256:" + "2" * 64},
                "entries": entries,
            }
        )
        build_inputs = {"suite": "libero_spatial", "task_id": 0}
        argv = [
            "--suites",
            "spatial",
            "--task-ids",
            "0",
            "--num-trials",
            "20",
            "--trial-batch-size",
            "6",
            "--manifest-dir",
            str(cache),
        ]
    publish_manifest(cache, runtime.manifest_cache_key(build_inputs), manifest)
    monkeypatch.setattr(runtime, "resolve_policy_model_spec", lambda argv: {"identity": "model"})
    batches = runtime.build_task_inputs(runtime.parse_submit_args(argv))
    assert len(batches) == 4
    assert sum(batch["num_episodes" if benchmark == "robotwin" else "num_trials"] for batch in batches) == 20
    assert {batch["manifest_hash"] for batch in batches} == {manifest["manifest_hash"]}


def test_official_totals_and_task_subsets_are_submission_inputs(tmp_path):
    robotwin = load_runtime("robotwin")
    robotwin_args = robotwin.parse_submit_args(
        [
            "--operation",
            "build_manifest",
            "--mode",
            "demo_clean",
            "--tasks",
            "adjust_bottle,click_bell",
            "--manifest-dir",
            str(tmp_path / "robotwin"),
        ]
    )
    robotwin_inputs = robotwin.build_task_inputs(robotwin_args)
    assert [item["task"] for item in robotwin_inputs] == ["adjust_bottle", "click_bell"]
    assert {item["total_episodes"] for item in robotwin_inputs} == {100}

    libero = load_runtime("libero")
    libero_args = libero.parse_submit_args(
        [
            "--operation",
            "build_manifest",
            "--suites",
            "spatial",
            "--task-ids",
            "0,3",
            "--manifest-dir",
            str(tmp_path / "libero"),
        ]
    )
    libero_inputs = libero.build_task_inputs(libero_args)
    assert [item["task_id"] for item in libero_inputs] == [0, 3]
    assert {item["total_trials"] for item in libero_inputs} == {50}


def test_client_result_must_match_current_task_manifest(tmp_path):
    runtime = load_runtime("libero")
    payload = {
        "suite": "libero_spatial",
        "task_id": 0,
        "trial_start": 5,
        "trial_stop": 6,
        "num_trials": 1,
        "successes": 1,
        "trials": [{"trial": 5, "init_state_index": 5, "success": True, "policy_steps": 1, "last_reward": 1.0}],
        "manifest_hash": "expected-manifest",
    }
    result = tmp_path / "results.json"
    result.write_text(json.dumps(payload))
    expected = dict(manifest_hash="expected-manifest")
    job, batch = runtime.BenchmarkTask("libero_spatial", 0), runtime.TrialBatch(5, 1)
    assert runtime._read_valid_result(result, job, batch, **expected) == payload
    payload["manifest_hash"] = "another-manifest"
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent"):
        runtime._read_valid_result(result, job, batch, **expected)


@pytest.mark.parametrize("benchmark", ["robotwin", "libero"])
def test_submit_argument_separator(benchmark):
    runtime = load_runtime(benchmark)
    options = ["--mode", "demo_clean"] if benchmark == "robotwin" else []
    deploy_args = ["--ckpt-dir", "/shared/model", "inference.denoise_steps=7"]
    args = runtime.parse_submit_args([*options, "--", *deploy_args])
    assert args.operation == "run_eval"
    assert args.deploy_args == deploy_args
    for invalid in (["--modle", "x"], deploy_args, ["--model", *deploy_args]):
        with pytest.raises(SystemExit) as error:
            runtime.parse_submit_args([*options, *invalid])
        assert error.value.code == 2
    args = runtime.parse_submit_args([*options, "--operation", "build_manifest", "--", *deploy_args])
    with pytest.raises(ValueError, match="only accepted for run_eval"):
        runtime.build_task_inputs(args)


@pytest.mark.parametrize("benchmark,base_port", [("robotwin", 8848), ("libero", 8920)])
@pytest.mark.parametrize("gpu,explicit_port", [(0, None), (3, None), (3, 12345)])
def test_worker_port_selection(benchmark, base_port, gpu, explicit_port, tmp_path, monkeypatch):
    runtime = load_runtime(benchmark)
    monkeypatch.setitem(sys.modules, "labtasker_runtime", runtime)
    path = ROOT / "benchmarks" / benchmark / "labtasker_worker.py"
    spec = importlib.util.spec_from_file_location(f"{benchmark}_worker_test", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    (tmp_path / "script").mkdir()
    (tmp_path / "script/eval_policy.py").touch()
    if benchmark == "libero":
        monkeypatch.setattr(runtime, "_check_worker_environment", lambda args: None)
    selected = []

    class Policy:
        def __init__(self, args):
            selected.append(args.port)

        def close(self):
            pass

    monkeypatch.setattr(worker, "WorkerPolicyServer", Policy)
    monkeypatch.setattr(worker, "run_worker", lambda *args: None)
    argv = ["--gpu", str(gpu), f"--{benchmark}-path", str(tmp_path), f"--{benchmark}-python", sys.executable]
    if explicit_port is not None:
        argv += ["--port", str(explicit_port)]
    assert worker.main(argv) == 0
    assert selected == [base_port + gpu if explicit_port is None else explicit_port]

"""Manual native-Windows smoke using public sources and synthetic requests only."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = "lr,svm,gbdt,kmeans"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def local_cli(argv: list[str], request=None) -> tuple[int, dict]:
    from student_tools.__main__ import main

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(argv, request=request)
    result = json.loads(output.getvalue())
    require(isinstance(result, dict), "CLI must return a JSON object")
    return code, result


class MemoryRequests:
    """Synthetic receipt store; never executes submissions or opens sockets."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, bytes]] = []
        self.receipts: dict[str, tuple[dict, bytes, dict]] = {}
        self.unavailable_once = False

    def __call__(self, operation, fields, *, payload=None):
        require(operation == "submit", "only synthetic submit is permitted")
        require(isinstance(payload, bytes), "submission payload must be bytes")
        fields = dict(fields)
        self.calls.append((fields, payload))
        if self.unavailable_once:
            self.unavailable_once = False
            return {"ok": False, "error": {"code": "SERVICE_UNAVAILABLE"}}
        key = fields["idempotency_key"]
        if key in self.receipts:
            prior_fields, prior_payload, receipt = self.receipts[key]
            if prior_fields != fields or prior_payload != payload:
                return {"ok": False, "error": {"code": "CONFLICT"}}
            return receipt
        data = {
            "kind": "submission_receipt",
            "submission_id": "1" * 32,
            "received_at": "2026-10-14T00:00:00.000000+00:00",
            "ai_declaration": fields["ai_declaration"],
            "data_version": "synthetic-windows-smoke-v1",
            "eval_config_id": "synthetic-config-v1",
            "status": "QUEUED",
            "attempt": 0,
        }
        if "selected_tasks" in fields:
            data["selected_tasks"] = fields["selected_tasks"]
        receipt = {"ok": True, "data": data}
        self.receipts[key] = (fields, payload, receipt)
        return receipt


def child_prepare(archive: str, store: str, barrier: str, quota: str) -> int:
    from student_tools import intents

    if quota != "default":
        intents.MAX_INTENTS = int(quota)
    deadline = time.monotonic() + 30
    while not Path(barrier).exists():
        require(time.monotonic() < deadline, "child barrier timed out")
        time.sleep(0.02)
    from student_tools.__main__ import main

    return main([
        "prepare", "--archive", archive, "--ai", "independent",
        "--tasks", TASKS, "--store", store,
    ])


def concurrent_prepare(archive: Path, store: Path, directory: Path,
                       *, last_slot: bool = False) -> None:
    barrier = directory / ("quota barrier" if last_slot else "start barrier")
    argv = [
        sys.executable, "-B", str(Path(__file__).resolve()),
        "--child-prepare", str(archive), str(store), str(barrier),
        "1" if last_slot else "default",
    ]
    processes = []
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(
                argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",
            ))
        barrier.write_text("start", encoding="ascii")
        keys = []
        rejected = 0
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            require(not stderr.strip(), "concurrent prepare wrote stderr")
            result = json.loads(stdout)
            if process.returncode == 0 and result.get("ok") is True:
                keys.append(result["data"]["idempotency_key"])
            else:
                require(last_slot and process.returncode == 1 and result == {
                    "ok": False, "error": {"code": "LOCAL_STATE_ERROR"},
                }, "unexpected concurrent prepare failure")
                rejected += 1
        expected = 1 if last_slot else 2
        require(len(keys) == len(set(keys)) == expected,
                "concurrent prepare success/key count drift")
        require(rejected == (1 if last_slot else 0), "quota rejection count drift")
        for key in keys:
            code, result = local_cli(["intent", key, "--store", str(store)])
            require(code == 0 and result["ok"], "concurrent intent not visible")
        from student_tools.intents_windows import DATABASE

        with contextlib.closing(sqlite3.connect(store / DATABASE)) as connection, connection:
            count = connection.execute("SELECT count(*) FROM intents").fetchone()[0]
        require(count == expected, "concurrent store row count drift")
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


def expect_intent_error(operation) -> None:
    from student_tools.intents import IntentError

    try:
        operation()
    except IntentError as error:
        require(str(error) == "invalid or unreadable student intent",
                "intent error leaked details")
    else:
        raise AssertionError("unsafe intent operation was accepted")


def storage_boundaries(payload: bytes, directory: Path, store: Path,
                       key: str, receipt: dict) -> None:
    from student_tools import intents
    from student_tools.intents_windows import DATABASE

    def create(root):
        return intents.create_intent(
            root, payload=payload, ai_declaration="independent",
            selected_tasks=TASKS.split(","),
        )

    def record(root, intent_key, submission_id=receipt["submission_id"]):
        return intents.record_receipt(
            root, intent_key, submission_id=submission_id,
            received_at=receipt["received_at"],
        )

    persisted = intents.read_intent(store, key)
    require(persisted.payload == payload, "stored archive bytes changed")
    require(persisted.selected_tasks == tuple(TASKS.split(",")), "stored tasks changed")
    require(persisted.ai_declaration == "independent", "stored declaration changed")
    replay = record(store, key)
    require(replay == persisted.receipt, "local receipt same-value replay drift")
    expect_intent_error(lambda: record(store, key, "2" * 32))
    require(intents.read_intent(store, key) == persisted, "receipt conflict changed record")

    quota_root = directory / "byte quota store"
    first = create(quota_root)
    with contextlib.closing(sqlite3.connect(quota_root / DATABASE)) as connection, connection:
        total = connection.execute(
            "SELECT sum(length(payload)+length(manifest)) FROM intents"
        ).fetchone()[0]
    original_limit = intents.MAX_TOTAL_INTENT_BYTES
    try:
        intents.MAX_TOTAL_INTENT_BYTES = total
        expect_intent_error(lambda: create(quota_root))
        expect_intent_error(lambda: record(quota_root, first.idempotency_key))
        require(intents.read_intent(quota_root, first.idempotency_key).receipt is None,
                "failed receipt quota wrote a receipt")
        cold_root = directory / "rollback retry store"
        intents.MAX_TOTAL_INTENT_BYTES = 1
        expect_intent_error(lambda: create(cold_root))
    finally:
        intents.MAX_TOTAL_INTENT_BYTES = original_limit
    require(create(cold_root).payload == payload, "failed cold create cannot recover")
    require(record(quota_root, first.idempotency_key) is not None, "receipt quota recovery failed")

    broken_root = directory / "bad database"
    broken_root.mkdir()
    (broken_root / DATABASE).write_bytes(b"not a SQLite database")
    expect_intent_error(lambda: create(broken_root))
    require((broken_root / DATABASE).read_bytes() == b"not a SQLite database",
            "bad database was overwritten")

    unknown_root = directory / "unknown schema"
    unknown_root.mkdir()
    with contextlib.closing(sqlite3.connect(unknown_root / DATABASE)) as connection, connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('synthetic preserved')")
    expect_intent_error(lambda: create(unknown_root))
    with contextlib.closing(sqlite3.connect(unknown_root / DATABASE)) as connection, connection:
        require(connection.execute("SELECT value FROM unrelated").fetchall() == [
            ("synthetic preserved",),
        ], "unknown schema data changed")
        require(connection.execute("PRAGMA user_version").fetchone()[0] == 0,
                "unknown schema version changed")

    corruptions = (
        ("payload digest", "payload", b"wrong ZIP bytes"),
        ("manifest shape", "manifest", b"{}"),
        ("manifest length", "manifest", b"x" * (intents.MAX_MANIFEST_BYTES + 1)),
        ("payload type", "payload", "not a BLOB"),
        ("receipt shape", "receipt", b"{}"),
        ("receipt length", "receipt", b"x" * (intents.MAX_RECEIPT_BYTES + 1)),
    )
    for label, column, value in corruptions:
        root = directory / label
        intent = create(root)
        with contextlib.closing(sqlite3.connect(root / DATABASE)) as connection, connection:
            connection.execute(f"UPDATE intents SET {column}=? WHERE key=?", (value, intent.idempotency_key))
        expect_intent_error(lambda: intents.read_intent(root, intent.idempotency_key))
        expect_intent_error(lambda: record(root, intent.idempotency_key))
        code, result = local_cli(["intent", intent.idempotency_key, "--store", str(root)])
        require(code == 1 and result == {
            "ok": False, "error": {"code": "LOCAL_STATE_ERROR"},
        }, "corrupt record CLI must fail closed")

    linked_root = directory / "hardlinked database"
    linked = create(linked_root)
    alias = directory / "database hardlink"
    os.link(linked_root / DATABASE, alias)
    expect_intent_error(lambda: intents.read_intent(linked_root, linked.idempotency_key))
    alias.unlink()
    require(intents.read_intent(linked_root, linked.idempotency_key).payload == payload,
            "hardlink rejection changed store")
    symbolic_root = directory / "symbolic store"
    try:
        symbolic_root.symlink_to(linked_root, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        print("SYMLINK_BOUNDARY_NOT_RUN: Windows privilege unavailable")
    else:
        expect_intent_error(lambda: intents.read_intent(symbolic_root, linked.idempotency_key))
        symbolic_root.unlink()
    print("REAL_STORAGE_BOUNDARIES_OK")

def selftest_environment_boundaries(directory: Path) -> None:
    from student_tools import selftest

    missing = directory / "missing executable.exe"
    require(not missing.exists(), "missing executable fixture exists")
    try:
        selftest.run_binary(missing, "", 5, str(directory))
    except selftest.EnvironmentError_:
        pass
    else:
        raise AssertionError("missing executable did not raise environment error")
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        code = selftest.main([
            "--project", str(ROOT / "starter"), "--compiler", str(missing),
            "--tasks", TASKS,
        ])
    require(code == 3 and "selftest: environment error:" in errors.getvalue(),
            "missing compiler CLI did not return environment exit 3")
    print("SELFTEST_ENVIRONMENT_BOUNDARIES_OK: no C++ executed")


def trusted_selftest() -> None:
    compiler = next((path for name in ("g++", "clang++")
                     if (path := shutil.which(name))), None)
    if compiler is None:
        print("TRUSTED_CPP_SELFTEST_NOT_RUN: no installed g++ or clang++")
        return
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=15, check=False,
    )
    require(version.returncode == 0, "installed compiler version check failed")
    lines = version.stdout.splitlines()
    require(bool(lines), "compiler version output missing")
    print("ACTUAL_COMPILER", lines[0][:250])
    from student_tools.selftest import compile_project

    with tempfile.TemporaryDirectory(prefix="trusted compile ") as workdir:
        executable, error = compile_project(compiler, str(ROOT / "starter"), workdir, 120)
        require(executable is not None, "trusted public starter compile failed")
        require(Path(executable).is_file(), "trusted compiler produced no executable")
        require(Path(executable).suffix == ".exe", "native executable suffix missing")
    print("TRUSTED_CPP_COMPILE_OK: one compile; placeholder quality NOT_RUN")


def run_smoke() -> None:
    require(os.name == "nt", "native Windows required; no platform simulation")
    sys.path.insert(0, str(ROOT))
    os.environ.update({
        "STUDENT_ID": "synthetic-windows-001",
        "STUDENT_NAME": "Synthetic Student",
        "STUDENT_NICKNAME": "Synthetic Smoke",
        "PYTHONUTF8": "1",
    })
    from student_tools.package import build_package, validate_package

    with tempfile.TemporaryDirectory(prefix="windows smoke ") as temporary:
        directory = Path(temporary) / "中文 path with spaces"
        directory.mkdir()
        project = directory / "trusted public starter"
        for name in (
            "README.md", "AI_USAGE.md", "include/assignment.hpp", "src/assignment.cpp",
        ):
            source = ROOT / "starter" / name
            require(source.is_file(), "required public starter member missing")
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        archive = directory / "source package.zip"
        build_package(project, archive)
        validate_package(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        code, config = local_cli(["config"])
        require(code == 0 and config["configured"], "synthetic config failed")
        require(config["authenticated"] is False, "config must not authenticate")
        store = directory / "local intent store"
        code, prepared = local_cli([
            "prepare", "--archive", str(archive), "--ai", "independent",
            "--tasks", TASKS, "--store", str(store),
        ])
        require(code == 0 and prepared["ok"], "prepare failed")
        intent = prepared["data"]
        require(intent["source_sha256"] == digest, "source hash changed")
        key = intent["idempotency_key"]
        code, reread = local_cli(["intent", key, "--store", str(store)])
        require(code == 0 and reread["data"] == intent, "intent reread drift")
        request = MemoryRequests()
        request.unavailable_once = True
        code, uncertain = local_cli(["retry", key, "--store", str(store)], request)
        require(code == 1 and not uncertain["ok"], "unavailable must fail closed")
        code, accepted = local_cli(["retry", key, "--store", str(store)], request)
        require(code == 0 and accepted["ok"], "synthetic retry failed")
        require(accepted["idempotency_key"] == key, "retry changed original key")
        code, replayed = local_cli(["retry", key, "--store", str(store)], request)
        require(code == 0 and replayed == accepted, "receipt replay drift")
        require(all(fields["idempotency_key"] == key for fields, _ in request.calls),
                "request changed retry key")
        require(all(payload == archive.read_bytes() for _, payload in request.calls),
                "request changed retry bytes")
        fields, payload = request.calls[-1]
        conflict = request("submit", fields, payload=payload + b"conflict")
        require(conflict == {"ok": False, "error": {"code": "CONFLICT"}},
                "synthetic conflict response drift")
        code, invalid = local_cli(["intent", "not-an-intent-key", "--store", str(store)])
        require(code == 1 and invalid == {
            "ok": False, "error": {"code": "LOCAL_STATE_ERROR"},
        }, "invalid intent must fail closed")
        storage_boundaries(archive.read_bytes(), directory, store, key, accepted["data"])
        concurrent_prepare(archive, directory / "parallel store", directory)
        concurrent_prepare(archive, directory / "quota race store", directory, last_slot=True)
        selftest_environment_boundaries(directory)
    trusted_selftest()
    print("PUBLIC_SYNTHETIC_WINDOWS_SMOKE_OK")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if len(sys.argv) == 6 and sys.argv[1] == "--child-prepare":
        require(os.name == "nt", "native Windows child required")
        raise SystemExit(child_prepare(*sys.argv[2:]))
    require(len(sys.argv) == 1, "unexpected smoke arguments")
    run_smoke()

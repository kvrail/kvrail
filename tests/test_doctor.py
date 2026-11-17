"""Tests for `kvrail doctor`.

Every check runs against a fake runner that returns recorded output, so the tests never touch the
host. Fixtures under fixtures/ are real outputs from a Mac Studio M3 Ultra on macOS 27.2.1 with
identifiers zeroed; the "peer_80" fixture adds a connected Mac mini to one bus.
"""

import ast
import json
import pathlib
from typing import Dict, List

import pytest

from kvrail import cli
from kvrail import doctor as d

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def runner_from(outputs: Dict[str, d.CommandResult]) -> d.Runner:
    """Return a runner that answers by command name and reports any other command as missing."""

    def runner(argv: List[str]) -> d.CommandResult:
        return outputs.get(argv[0], d.CommandResult(d.Outcome.not_found))

    return runner


def exited(output: str, exit_code: int = 0) -> d.CommandResult:
    return d.CommandResult(d.Outcome.exited, exit_code, output)


def test_macos_version_boundaries():
    # Goal: the 26.2 boundary holds on both sides, and only canonical versions parse.
    assert d.parse_macos_version("26.2") == (26, 2)
    assert d.parse_macos_version("27.0\n") == (27, 0)
    assert d.parse_macos_version("26") == (26, 0)
    assert d.parse_macos_version("26.1.9") < d.MACOS_VERSION_MIN
    assert d.parse_macos_version("26.2.1") >= d.MACOS_VERSION_MIN
    for bad in ["", "garbage", "26.2.1.5", "99999.1", "26.", "v26.2", "26.2 beta"]:
        assert d.parse_macos_version(bad) is None, bad


def test_rdma_status_disabled_is_not_enabled():
    # Goal: "disabled" contains "enabled" as a substring; it must still read as disabled.
    assert d.parse_rdma_status("disabled\n") is False
    assert d.parse_rdma_status("enabled\n") is True
    assert d.parse_rdma_status("RDMA is not enabled") is None
    checks = d.check_rdma(runner_from({"rdma_ctl": exited("disabled\n")}))
    assert [(c.id, c.status, c.detail) for c in checks] == [("rdma_ctl", d.Status.fail, "disabled")]


def test_rdma_enabled_device_states():
    # Goal: a missing ibv_devices is a warning, an empty list is a failure, devices are ok.
    def statuses(ibv: d.CommandResult) -> List[d.Status]:
        runner = runner_from({"rdma_ctl": exited("enabled\n"), "ibv_devices": ibv})
        return [check.status for check in d.check_rdma(runner)]

    assert statuses(d.CommandResult(d.Outcome.not_found)) == [d.Status.ok, d.Status.warn]
    assert statuses(exited(fixture("ibv_devices_empty.txt"))) == [d.Status.ok, d.Status.fail]
    listed = "    device   node GUID\n    ------   ----\n    rdma_en2  0000\n    rdma_en3\n"
    assert d.parse_ibv_devices(listed) == ["rdma_en2", "rdma_en3"]
    assert statuses(exited(listed)) == [d.Status.ok, d.Status.ok]
    assert statuses(exited(listed, exit_code=1)) == [d.Status.ok, d.Status.warn]
    assert d.parse_ibv_devices("    rdma_enx 0\n    rdma_en2x 0\n") == []


def test_rdma_ctl_missing_or_hung():
    # Goal: operating errors of rdma_ctl become failures, never exceptions.
    assert d.check_rdma(runner_from({}))[0].status == d.Status.fail
    hung = runner_from({"rdma_ctl": d.CommandResult(d.Outcome.timeout)})
    assert d.check_rdma(hung)[0].detail == "rdma_ctl timed out after 15 s"
    broken = runner_from({"rdma_ctl": d.CommandResult(d.Outcome.failed)})
    assert d.check_rdma(broken)[0].detail == "cannot run rdma_ctl"


def test_thunderbolt_idle_ports_are_not_a_link():
    # Goal: the regression. Idle ports report "Up to 120 Gb/s"; that is not a TB5 link.
    output = fixture("thunderbolt_mac_studio_no_peer.json")
    [check] = d.check_thunderbolt(runner_from({"system_profiler": exited(output)}))
    assert check.status == d.Status.warn
    assert check.detail == "6 ports, no peer connected"


def test_thunderbolt_connected_peer_at_80():
    # Goal: a connected local port at 80 Gb/s passes; the peer's own ports are not counted.
    output = fixture("thunderbolt_mac_studio_peer_80.json")
    ports = d.parse_thunderbolt_ports(output)
    assert len(ports) == 6
    [check] = d.check_thunderbolt(runner_from({"system_profiler": exited(output)}))
    assert check.status == d.Status.ok
    assert check.detail == "6 ports, connected: receptacle 1 at 80 Gb/s"


def test_thunderbolt_slow_link_and_nested_peer():
    # Goal: a 40 Gb/s link warns, and an 80 Gb/s port of a nested peer does not rescue it.
    document = json.loads(fixture("thunderbolt_mac_studio_peer_80.json"))
    bus = next(bus for bus in document["SPThunderboltDataType"] if "_items" in bus)
    bus["receptacle_1_tag"]["current_speed_key"] = "40 Gb/s"
    [check] = d.check_thunderbolt(runner_from({"system_profiler": exited(json.dumps(document))}))
    assert check.status == d.Status.warn
    assert "40 Gb/s" in check.detail

    bus["receptacle_1_tag"]["receptacle_status_key"] = "receptacle_no_devices_connected"
    [check] = d.check_thunderbolt(runner_from({"system_profiler": exited(json.dumps(document))}))
    assert check.detail == "6 ports, no peer connected"


def test_thunderbolt_malformed_output():
    # Goal: every malformed shape is a warning, including deep nesting and too many ports.
    too_many = {"SPThunderboltDataType": [{f"receptacle_{n}_tag": {}} for n in range(65)]}
    for output in ["{", "[]", '{"SPThunderboltDataType": {}}', '{"SPThunderboltDataType": [1]}',
                   '{"SPThunderboltDataType": [{"receptacle_1_tag": {"current_speed_key": 1}}]}',
                   "[" * 100_000 + "]" * 100_000, json.dumps(too_many)]:
        [check] = d.check_thunderbolt(runner_from({"system_profiler": exited(output)}))
        assert check.status == d.Status.warn, output[:60]
    [check] = d.check_thunderbolt(runner_from({"system_profiler": exited("", 1)}))
    assert (check.status, check.detail) == (d.Status.skip, "exit code 1")
    hung = runner_from({"system_profiler": d.CommandResult(d.Outcome.timeout)})
    assert d.check_thunderbolt(hung)[0].detail == "system_profiler timed out after 15 s"


def test_bridge_detection():
    # Goal: the bridge is found by hardware port, so a renamed service is still detected.
    real = fixture("service_order_bridge_enabled.txt")
    assert d.check_bridge(runner_from({"networksetup": exited(real)}))[0].status == d.Status.warn
    renamed = real.replace("(3) Thunderbolt Bridge", "(*) TB fabric")
    [check] = d.check_bridge(runner_from({"networksetup": exited(renamed)}))
    assert (check.status, check.detail) == (d.Status.ok, "disabled")
    no_bridge = d.SERVICE_ORDER_HEADER + "\n(1) Wi-Fi\n(Hardware Port: Wi-Fi, Device: en1)\n"
    [check] = d.check_bridge(runner_from({"networksetup": exited(no_bridge)}))
    assert check.detail == "not present"
    services = d.parse_service_order(real)
    assert [(s.name, s.device, s.enabled) for s in services] == [
        ("Ethernet", "en0", True), ("Wi-Fi", "en1", True), ("Thunderbolt Bridge", "bridge0", True),
    ]


def test_bridge_unknown_output_is_not_absent():
    # Goal: output that cannot be read leaves the bridge state unknown (skip), never "not present".
    real = fixture("service_order_bridge_enabled.txt")
    incomplete = real.replace("(Hardware Port: Thunderbolt Bridge, Device: bridge0)\n", "")
    for output in ["", "\n", "garbage\n", real.split("\n", 1)[1], incomplete]:
        [check] = d.check_bridge(runner_from({"networksetup": exited(output)}))
        assert (check.status, check.detail) == (d.Status.skip, "cannot parse networksetup output")


def test_hardware_rejects_unbounded_numbers():
    # Goal: sysctl values parse only as bounded unsigned integers.
    runner = runner_from({"sysctl": exited("9" * 21)})
    assert d.check_hardware(runner) == [d.Check("chip", "Chip", d.Status.info, "9" * 21)]
    runner = runner_from({"sysctl": exited("549755813888\n")})
    details = [check.detail for check in d.check_hardware(runner)]
    assert details == ["549755813888", "512 GiB", "549755813888 MiB"]


def test_exit_codes():
    # Goal: warnings fail only in strict mode; failures always fail.
    warn = [d.Check("a", "A", d.Status.warn)]
    fail = [d.Check("a", "A", d.Status.info), d.Check("b", "B", d.Status.fail)]
    assert d.exit_code(warn, strict=False) == 0
    assert d.exit_code(warn, strict=True) == 1
    skip = [d.Check("a", "A", d.Status.skip)]
    assert d.exit_code(skip, strict=False) == 0
    assert d.exit_code(skip, strict=True) == 1
    assert d.exit_code(fail, strict=False) == 1


def test_check_invariants():
    # Goal: a hint on a passing check is a programmer error and must stop the program.
    with pytest.raises(AssertionError):
        d.Check("a", "A", d.Status.ok, "", "hint")
    with pytest.raises(AssertionError):
        d.Check("", "A", d.Status.ok)


def test_render_cleans_command_output():
    # Goal: control characters from a command cannot reach the terminal, and length is bounded.
    hostile = "\x1b[2J\x9b" + "x" * 500
    text = d.render_text([d.Check("a", "A", d.Status.warn, hostile, "fix")])
    assert "\x1b" not in text
    assert text.splitlines() == ["[warn] A: ?[2J?" + "x" * 192 + "...", "       -> fix"]
    [obj] = json.loads(d.render_json([d.Check("a", "A", d.Status.ok, 'q"\\\n')]))
    assert obj == {"id": "a", "title": "A", "status": "ok", "detail": 'q"\\?', "hint": ""}


def test_run_checks_off_macos(monkeypatch):
    # Goal: off macOS only the platform check runs, and it fails.
    monkeypatch.setattr(d.platform, "system", lambda: "Linux")
    checks = d.run_checks(runner_from({}))
    assert [(c.id, c.status) for c in checks] == [("platform", d.Status.fail)]


def test_cli_usage_errors(capsys):
    # Goal: scripts can tell a usage error (2) from a passing (0) or failing (1) check.
    assert cli.main([]) == 2
    for argv in [["bogus"], ["doctor", "--bogus"]]:
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2


def test_run_command_bounds(monkeypatch):
    # Goal: the real runner reports a missing binary and enforces the output limit.
    monkeypatch.setitem(d.COMMAND_PATHS, "sw_vers", "/nonexistent/sw_vers")
    assert d.run_command(["sw_vers"]).outcome == d.Outcome.not_found
    # A path that exists but cannot run is an operating error, not a traceback.
    monkeypatch.setitem(d.COMMAND_PATHS, "sw_vers", str(FIXTURES / "ibv_devices_empty.txt"))
    assert d.run_command(["sw_vers"]).outcome == d.Outcome.failed
    monkeypatch.setitem(d.COMMAND_PATHS, "sysctl", "/usr/bin/yes")
    assert d.run_command(["sysctl"]).outcome == d.Outcome.output_too_large
    monkeypatch.setitem(d.COMMAND_PATHS, "networksetup", "/bin/echo")
    assert d.run_command(["networksetup", "hi"]) == exited("hi\n")
    # A child that closes stdout but keeps running must still hit the timeout.
    monkeypatch.setattr(d, "COMMAND_TIMEOUT_S", 0.5)
    monkeypatch.setitem(d.COMMAND_PATHS, "ibv_devices", "/bin/sh")
    assert d.run_command(["ibv_devices", "-c", "exec >&-; sleep 5"]).outcome == d.Outcome.timeout


def test_style_limits():
    # Goal: enforce the profile's 100-column lines and 70-line functions in the build.
    for path in sorted((ROOT / "src").rglob("*.py")) + [pathlib.Path(__file__)]:
        source = path.read_text()
        for number, line in enumerate(source.splitlines(), start=1):
            assert len(line) <= 100, f"{path.name}:{number} is {len(line)} columns"
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                assert length <= 70, f"{path.name}:{node.name} is {length} lines"

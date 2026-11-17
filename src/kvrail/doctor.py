"""`kvrail doctor`: checks that a Mac is ready for RDMA over Thunderbolt 5.

Standard library only. The output of every external command is untrusted input: a missing
binary, a timeout, oversized output or an unknown format is an operating error and becomes a
"skip", "warn" or "fail" check, never an exception. `invariant()` guards programmer errors only.
See docs/style-profile.md for the limits and the reasons behind them.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

# RDMA over Thunderbolt ships with macOS 26.2 (Apple TN3205).
MACOS_VERSION_MIN: Tuple[int, int] = (26, 2)

# `system_profiler` is the slowest command and takes about 1 s on a Mac Studio.
COMMAND_TIMEOUT_S = 15.0
COMMAND_POLL_S = 0.01
# `system_profiler -json` for six idle ports is about 3 KiB, so 1 MiB is far beyond normal.
COMMAND_OUTPUT_BYTES_MAX = 1024 * 1024

# A Mac Studio has six Thunderbolt receptacles. The limits leave room for future machines.
THUNDERBOLT_PORT_COUNT_MAX = 64
NETWORK_SERVICE_COUNT_MAX = 64
RDMA_DEVICE_COUNT_MAX = 64
# Thunderbolt 5 runs at 80 Gb/s symmetric, or 120 Gb/s with Bandwidth Boost.
THUNDERBOLT_5_SPEED_GBPS_MIN = 80
THUNDERBOLT_BRIDGE_HARDWARE_PORT = "Thunderbolt Bridge"
THUNDERBOLT_BRIDGE_DEVICE = "bridge0"
# The first line of `networksetup -listnetworkserviceorder`. Without it the format is unknown.
SERVICE_ORDER_HEADER = "An asterisk (*) denotes that a network service is disabled."
DETAIL_CHARS_MAX = 200

# Commands run from absolute paths so that PATH cannot redirect them.
COMMAND_PATHS: Dict[str, str] = {
    "sw_vers": "/usr/bin/sw_vers",
    "sysctl": "/usr/sbin/sysctl",
    "rdma_ctl": "/usr/bin/rdma_ctl",
    "ibv_devices": "/usr/bin/ibv_devices",
    "system_profiler": "/usr/sbin/system_profiler",
    "networksetup": "/usr/sbin/networksetup",
}


def invariant(condition: bool, message: str = "invariant violated") -> None:
    """Assert a programmer invariant. Unlike `assert`, `python -O` does not remove it."""
    if not condition:
        raise AssertionError(message)


invariant(MACOS_VERSION_MIN >= (26, 2))
invariant(0 < COMMAND_POLL_S < COMMAND_TIMEOUT_S <= 60)
invariant(THUNDERBOLT_5_SPEED_GBPS_MIN == 80)


class Status(str, Enum):
    ok = "ok"
    warn = "warn"
    fail = "fail"
    skip = "skip"
    info = "info"


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    status: Status
    detail: str = ""
    hint: str = ""

    def __post_init__(self) -> None:
        invariant(bool(self.id), "check id is empty")
        invariant(bool(self.title), "check title is empty")
        invariant(isinstance(self.status, Status), "check status is not a Status")
        # A hint tells the operator what to fix, so it exists only when something is wrong.
        if self.hint:
            invariant(self.status in (Status.warn, Status.fail), "hint on a passing check")


class Outcome(str, Enum):
    exited = "exited"
    not_found = "not found"
    timeout = "timed out"
    output_too_large = "output too large"
    failed = "failed"  # The command exists but could not start or be read.


@dataclass(frozen=True)
class CommandResult:
    outcome: Outcome
    exit_code: int = -1
    output: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome == Outcome.exited and self.exit_code == 0


@dataclass(frozen=True)
class ThunderboltPort:
    receptacle: str
    speed: str
    connected: bool


@dataclass(frozen=True)
class NetworkService:
    name: str
    hardware_port: str
    device: str
    enabled: bool


# Runs a command named in COMMAND_PATHS. Injected so that tests never touch the host.
Runner = Callable[[List[str]], CommandResult]


def run_command(argv: List[str]) -> CommandResult:
    invariant(argv[0] in COMMAND_PATHS, f"unknown command {argv[0]}")
    try:
        process = subprocess.Popen(
            [COMMAND_PATHS[argv[0]]] + argv[1:],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # Stderr carries warnings that would corrupt the JSON of `system_profiler`.
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return CommandResult(Outcome.not_found)
    except OSError:
        # Permission, ENOEXEC or descriptor exhaustion: an operating error, never a traceback.
        return CommandResult(Outcome.failed)

    chunks: List[bytes] = []
    reader = threading.Thread(target=run_command_read, args=(process, chunks), daemon=True)
    try:
        reader.start()
    except RuntimeError:
        process.kill()
        process.wait()
        return CommandResult(Outcome.failed)
    deadline = time.monotonic() + COMMAND_TIMEOUT_S
    while process.poll() is None and reader.is_alive():
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            return CommandResult(Outcome.timeout)
        time.sleep(COMMAND_POLL_S)
    reader.join(timeout=max(0.0, deadline - time.monotonic()))
    output = b"".join(chunks)
    if len(output) > COMMAND_OUTPUT_BYTES_MAX:
        process.kill()
        process.wait()
        return CommandResult(Outcome.output_too_large)
    if reader.is_alive():
        process.kill()
        process.wait()
        return CommandResult(Outcome.timeout)
    # Stdout can close before the process exits, so the final wait is bounded too.
    try:
        exit_code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return CommandResult(Outcome.timeout)
    return CommandResult(Outcome.exited, exit_code, output.decode("utf-8", errors="replace"))


def run_command_read(process: "subprocess.Popen[bytes]", chunks: List[bytes]) -> None:
    invariant(process.stdout is not None)
    # One byte past the limit is enough to know that the limit was crossed.
    chunks.append(process.stdout.read(COMMAND_OUTPUT_BYTES_MAX + 1))


def run_checks(runner: Runner = run_command) -> List[Check]:
    checks = check_platform(runner)
    if any(check.id == "platform" for check in checks):
        # The other checks call macOS-only commands and would only add noise.
        return checks
    checks += check_hardware(runner)
    checks += check_rdma(runner)
    checks += check_thunderbolt(runner)
    checks += check_bridge(runner)
    return checks


def exit_code(checks: List[Check], strict: bool) -> int:
    for check in checks:
        if check.status == Status.fail:
            return 1
        if strict and check.status == Status.warn:
            return 1
    return 0


def render_text(checks: List[Check]) -> str:
    tags: Dict[Status, str] = {
        Status.ok: "[ ok ]",
        Status.warn: "[warn]",
        Status.fail: "[FAIL]",
        Status.skip: "[skip]",
        Status.info: "[info]",
    }
    invariant(len(tags) == len(Status))
    lines: List[str] = []
    for check in checks:
        detail = clean_detail(check.detail)
        if detail:
            lines.append(f"{tags[check.status]} {check.title}: {detail}")
        else:
            lines.append(f"{tags[check.status]} {check.title}")
        if check.hint:
            lines.append(f"       -> {check.hint}")
    return "\n".join(lines)


def render_json(checks: List[Check]) -> str:
    objects = []
    for check in checks:
        fields = asdict(check)
        fields["status"] = check.status.value
        fields["detail"] = clean_detail(check.detail)
        objects.append(fields)
    return json.dumps(objects, indent=2)


def clean_detail(detail: str) -> str:
    """Make command output safe for a terminal: no control characters, bounded length."""
    cleaned = "".join("?" if is_control(char) else char for char in detail)
    if len(cleaned) > DETAIL_CHARS_MAX:
        cleaned = cleaned[: DETAIL_CHARS_MAX - 3] + "..."
    invariant(len(cleaned) <= DETAIL_CHARS_MAX)
    return cleaned


def is_control(char: str) -> bool:
    # C0, DEL and C1, the same set as Rust `char::is_control`. C1 holds the 8-bit CSI (U+009B).
    return ord(char) < 32 or 127 <= ord(char) < 160


def check_platform(runner: Runner) -> List[Check]:
    title = "macOS on Apple Silicon"
    if platform.system() != "Darwin":
        hint = "KVRail targets macOS 26.2+ on Apple Silicon."
        return [Check("platform", title, Status.fail, platform.platform(), hint)]
    if platform.machine() != "arm64":
        hint = "Intel Macs are not supported. Under Rosetta, use an arm64 Python."
        return [Check("platform", title, Status.fail, platform.machine(), hint)]

    result = runner(["sw_vers", "-productVersion"])
    version = parse_macos_version(result.output) if result.succeeded else None
    if version is None:
        return [Check("macos", "macOS version", Status.warn, "cannot read sw_vers")]
    title = "macOS >= 26.2 (RDMA over Thunderbolt)"
    if version >= MACOS_VERSION_MIN:
        return [Check("macos", title, Status.ok, result.output.strip())]
    hint = "Update to macOS 26.2 or newer on every node."
    return [Check("macos", title, Status.fail, result.output.strip(), hint)]


def check_hardware(runner: Runner) -> List[Check]:
    checks: List[Check] = []
    result = runner(["sysctl", "-n", "machdep.cpu.brand_string"])
    if result.succeeded and result.output.strip():
        checks.append(Check("chip", "Chip", Status.info, result.output.strip()))

    memory_bytes = parse_unsigned(runner(["sysctl", "-n", "hw.memsize"]))
    if memory_bytes is not None:
        memory_gib = memory_bytes // (1024 * 1024 * 1024)
        checks.append(Check("memory", "Unified memory", Status.info, f"{memory_gib} GiB"))

    # Zero means macOS picks the limit. Raise it with `sudo sysctl iogpu.wired_limit_mb=N`.
    wired_limit_mib = parse_unsigned(runner(["sysctl", "-n", "iogpu.wired_limit_mb"]))
    if wired_limit_mib is not None:
        if wired_limit_mib == 0:
            detail = "system default"
        else:
            detail = f"{wired_limit_mib} MiB"
        checks.append(Check("wired", "GPU wired memory limit", Status.info, detail))
    return checks


def check_rdma(runner: Runner) -> List[Check]:
    title = "RDMA enabled"
    hint_enable = "Boot into Recovery, open Terminal, run `rdma_ctl enable`, reboot."
    result = runner(["rdma_ctl", "status"])
    if result.outcome == Outcome.not_found:
        hint = "Needs macOS 26.2+. " + hint_enable
        return [Check("rdma_ctl", title, Status.fail, "rdma_ctl not found", hint)]
    if result.outcome != Outcome.exited:
        # The RDMA state is unknown, and an unknown state must never pass as ready.
        detail = command_problem("rdma_ctl", result)
        return [Check("rdma_ctl", title, Status.fail, detail, hint_enable)]
    enabled = parse_rdma_status(result.output)
    if enabled is None:
        detail = result.output.strip() or f"exit code {result.exit_code}"
        return [Check("rdma_ctl", title, Status.fail, detail, hint_enable)]
    if not enabled:
        return [Check("rdma_ctl", title, Status.fail, "disabled", hint_enable)]

    # Verbs devices exist only once RDMA is enabled, so they are checked only then.
    return [Check("rdma_ctl", title, Status.ok, "enabled"), check_rdma_devices(runner)]


def check_rdma_devices(runner: Runner) -> Check:
    title = "RDMA verbs devices"
    result = runner(["ibv_devices"])
    if result.outcome == Outcome.not_found:
        return Check("ibv", title, Status.warn, "ibv_devices not found")
    if result.outcome != Outcome.exited:
        return Check("ibv", title, Status.warn, command_problem("ibv_devices", result))
    try:
        devices = parse_ibv_devices(result.output)
    except ValueError:
        return Check("ibv", title, Status.warn, "cannot parse ibv_devices output")
    if devices:
        return Check("ibv", title, Status.ok, ", ".join(devices))
    hint = (
        "Connect a Thunderbolt 5 cable to a peer Mac (on Mac Studio avoid the port next to "
        "Ethernet), re-plug, and reboot if no rdma_enX appears."
    )
    return Check("ibv", title, Status.fail, "none listed", hint)


def command_problem(name: str, result: CommandResult) -> str:
    """Describe a command that produced no usable output, in the same words as the Rust CLI."""
    invariant(result.outcome != Outcome.exited)
    if result.outcome == Outcome.not_found:
        return f"{name} not found"
    if result.outcome == Outcome.timeout:
        return f"{name} timed out after {COMMAND_TIMEOUT_S:g} s"
    if result.outcome == Outcome.output_too_large:
        return f"{name} output is over 1 MiB"
    return f"cannot run {name}"


def check_thunderbolt(runner: Runner) -> List[Check]:
    title = "Thunderbolt 5 link (80/120 Gb/s)"
    result = runner(["system_profiler", "SPThunderboltDataType", "-json"])
    if not result.succeeded or not result.output.strip():
        return [Check("tb", title, Status.skip, "system_profiler unavailable")]
    try:
        ports = parse_thunderbolt_ports(result.output)
    except ValueError:
        return [Check("tb", title, Status.warn, "cannot parse system_profiler output")]
    if not ports:
        return [Check("tb", title, Status.warn, "no Thunderbolt ports reported")]

    links = [port for port in ports if port.connected]
    detail = check_thunderbolt_detail(ports, links)
    if not links:
        hint = "Connect a Thunderbolt 5 cable to the peer Mac."
        return [Check("tb", title, Status.warn, detail, hint)]
    for link in links:
        speed_gbps = parse_speed_gbps(link.speed)
        if speed_gbps is not None and speed_gbps >= THUNDERBOLT_5_SPEED_GBPS_MIN:
            return [Check("tb", title, Status.ok, detail)]
    hint = "Link is below 80 Gb/s: use a Thunderbolt 5 cable and Thunderbolt 5 ports on both Macs."
    return [Check("tb", title, Status.warn, detail, hint)]


def check_thunderbolt_detail(ports: List[ThunderboltPort], links: List[ThunderboltPort]) -> str:
    invariant(len(links) <= len(ports))
    if not links:
        return f"{len(ports)} ports, no peer connected"
    connected = ", ".join(f"receptacle {link.receptacle} at {link.speed}" for link in links)
    return f"{len(ports)} ports, connected: {connected}"


def check_bridge(runner: Runner) -> List[Check]:
    title = "Thunderbolt Bridge service"
    result = runner(["networksetup", "-listnetworkserviceorder"])
    if not result.succeeded:
        return [Check("bridge", title, Status.skip, "networksetup unavailable")]
    try:
        services = parse_service_order(result.output)
    except ValueError:
        return [Check("bridge", title, Status.skip, "cannot parse networksetup output")]
    for service in services:
        if not check_bridge_matches(service):
            continue
        if service.enabled:
            hint = (
                "With 2+ cables between the same Macs it can form a network loop. "
                "Disable it in System Settings -> Network."
            )
            return [Check("bridge", title, Status.warn, "enabled", hint)]
        return [Check("bridge", title, Status.ok, "disabled")]
    return [Check("bridge", title, Status.ok, "not present")]


def check_bridge_matches(service: NetworkService) -> bool:
    # The user can rename the service, so match on the hardware port and its device instead.
    if service.hardware_port == THUNDERBOLT_BRIDGE_HARDWARE_PORT:
        return True
    return service.device == THUNDERBOLT_BRIDGE_DEVICE


def parse_macos_version(output: str) -> Optional[Tuple[int, int]]:
    """Parse "26", "26.2" or "26.2.1". Anything else, including "26.2.1.5", is rejected."""
    match = re.fullmatch(r"(\d{1,4})(?:\.(\d{1,4}))?(?:\.\d{1,4})?", output.strip())
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or "0")
    invariant(0 <= major <= 9999)
    invariant(0 <= minor <= 9999)
    return major, minor


def parse_unsigned(result: CommandResult) -> Optional[int]:
    # Twenty digits hold any u64, so a Python int parsed here cannot grow without bound.
    if not result.succeeded:
        return None
    match = re.fullmatch(r"\d{1,20}", result.output.strip())
    if match is None:
        return None
    return int(match.group(0))


def parse_rdma_status(output: str) -> Optional[bool]:
    """Return True for "enabled", False for "disabled", None for anything else."""
    status = output.strip().lower()
    if status == "enabled":
        return True
    if status == "disabled":
        return False
    return None


def parse_ibv_devices(output: str) -> List[str]:
    devices = re.findall(r"^[ \t]*(rdma_en\d{1,4})(?:[ \t]|$)", output, re.MULTILINE)
    if len(devices) > RDMA_DEVICE_COUNT_MAX:
        raise ValueError("too many RDMA devices")
    return devices


def parse_service_order(output: str) -> List[NetworkService]:
    """Parse `networksetup -listnetworkserviceorder`.

    After the header line, each service is a "(1) Name" line, or "(*) Name" when disabled,
    followed by a "(Hardware Port: Port, Device: dev)" line. An unknown header, a service line
    without its hardware line, or too many services raise ValueError: the bridge state is then
    unknown, which must not be reported as "not present".
    """
    lines = [line.strip() for line in output.splitlines()]
    if not lines or lines[0] != SERVICE_ORDER_HEADER:
        raise ValueError("unknown networksetup header")
    services: List[NetworkService] = []
    pending: Optional[Tuple[str, bool]] = None
    for line in lines[1:]:
        if pending is not None:
            hardware = re.fullmatch(r"\(Hardware Port: (.*), Device: (.*)\)", line)
            if hardware is None:
                raise ValueError("network service without a hardware port line")
            if len(services) == NETWORK_SERVICE_COUNT_MAX:
                raise ValueError("too many network services")
            name, enabled = pending
            services.append(NetworkService(name, hardware.group(1), hardware.group(2), enabled))
            pending = None
            continue
        service = re.fullmatch(r"\((\d{1,4}|\*)\) (.+)", line)
        if service is not None:
            pending = (service.group(2), service.group(1) != "*")
    if pending is not None:
        raise ValueError("network service without a hardware port line")
    return services


def parse_thunderbolt_ports(output: str) -> List[ThunderboltPort]:
    """Parse the receptacles of the local Thunderbolt buses from `system_profiler -json`.

    Only the "receptacle_<N>_tag" objects of each top-level bus are ports of this Mac. A connected
    peer appears under the bus "_items" with its own receptacle tags, which are ignored.
    Raises ValueError when the output does not have this shape.
    """
    invariant(len(output) <= COMMAND_OUTPUT_BYTES_MAX)
    try:
        document = json.loads(output)
    except RecursionError as error:
        # `json.loads` recurses on nesting; the input is bounded, so this is a parse failure.
        raise ValueError("system_profiler output is nested too deeply") from error
    if not isinstance(document, dict):
        raise ValueError("system_profiler output is not an object")
    buses = document.get("SPThunderboltDataType")
    if not isinstance(buses, list):
        raise ValueError("SPThunderboltDataType is not a list")
    ports: List[ThunderboltPort] = []
    for bus in buses:
        if not isinstance(bus, dict):
            raise ValueError("Thunderbolt bus is not an object")
        for key in sorted(bus):
            receptacle = re.fullmatch(r"receptacle_(\d{1,4})_tag", key)
            if receptacle is None:
                continue
            if len(ports) == THUNDERBOLT_PORT_COUNT_MAX:
                raise ValueError("too many Thunderbolt ports")
            ports.append(parse_thunderbolt_port(receptacle.group(1), bus[key]))
    return ports


def parse_thunderbolt_port(receptacle: str, tag: object) -> ThunderboltPort:
    if not isinstance(tag, dict):
        raise ValueError("Thunderbolt receptacle is not an object")
    speed = tag.get("current_speed_key", "")
    status = tag.get("receptacle_status_key", "")
    if not isinstance(speed, str) or not isinstance(status, str):
        raise ValueError("Thunderbolt receptacle fields are not strings")
    # An idle port reports its capability ("Up to 120 Gb/s"), not a link speed.
    return ThunderboltPort(receptacle, speed, status == "receptacle_connected")


def parse_speed_gbps(speed: str) -> Optional[int]:
    match = re.search(r"(\d{1,4}) ?Gb/s", speed)
    if match is None:
        return None
    return int(match.group(1))

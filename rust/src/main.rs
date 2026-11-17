//! `kvrail doctor`: checks that a Mac is ready for RDMA over Thunderbolt 5.
//!
//! Zero dependencies. Every external command is an operating error source, not a programmer
//! error: a missing binary, a timeout, oversized output or an unknown format becomes a `skip`,
//! `warn` or `fail` check, never a panic. Assertions guard programmer errors only, and they use
//! `assert!` so that they also hold in release builds.

#![warn(clippy::too_many_lines)]

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::process::{Child, Command, ExitCode, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// RDMA over Thunderbolt ships with macOS 26.2 (Apple TN3205).
const MACOS_VERSION_MIN: [u32; 2] = [26, 2];
// Four digits hold every real version component and keep `u32` parsing far from overflow.
const VERSION_COMPONENT_DIGITS_MAX: usize = 4;
// Twenty digits hold `u64::MAX`, so no sysctl value can grow without bound.
const DECIMAL_DIGITS_MAX: usize = 20;

// `system_profiler` is the slowest command and takes about 1 s on a Mac Studio.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(15);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(10);
// The loop bound is the deadline in polls, so the loop ends even if the clock misbehaves.
const COMMAND_POLL_COUNT_MAX: u32 = 1500;
// `system_profiler` for six idle ports prints about 2 KiB. A larger output is not trustworthy.
const COMMAND_OUTPUT_BYTES_MAX: u64 = 1024 * 1024;

// A Mac Studio has six Thunderbolt receptacles. The limit leaves room for future machines.
const THUNDERBOLT_PORT_COUNT_MAX: usize = 64;
// Thunderbolt 5 runs at 80 Gb/s symmetric or 120 Gb/s with Bandwidth Boost.
const THUNDERBOLT_5_SPEED_GBPS_MIN: u32 = 80;
const NETWORK_SERVICE_COUNT_MAX: usize = 64;
const RDMA_DEVICE_COUNT_MAX: usize = 64;
const RDMA_DEVICE_DIGITS_MAX: usize = 4;
// A long detail would hide the verdict of the next checks in the terminal.
const DETAIL_CHARS_MAX: usize = 200;

// The bridge is matched by hardware port and device, because the user can rename the service.
const THUNDERBOLT_BRIDGE_HARDWARE_PORT: &str = "Thunderbolt Bridge";
const THUNDERBOLT_BRIDGE_DEVICE: &str = "bridge0";
/// The first line of `networksetup -listnetworkserviceorder`. Without it the format is unknown.
const SERVICE_ORDER_HEADER: &str = "An asterisk (*) denotes that a network service is disabled.";

// Absolute paths keep `PATH` from redirecting a command to another program.
const SW_VERS: &str = "/usr/bin/sw_vers";
const SYSCTL: &str = "/usr/sbin/sysctl";
const RDMA_CTL: &str = "/usr/bin/rdma_ctl";
const IBV_DEVICES: &str = "/usr/bin/ibv_devices";
const SYSTEM_PROFILER: &str = "/usr/sbin/system_profiler";
const NETWORKSETUP: &str = "/usr/sbin/networksetup";

const HINT_RDMA_ENABLE: &str = "Boot into Recovery, open Terminal, run `rdma_ctl enable`, reboot.";
const HINT_RDMA_NOT_FOUND: &str =
    "Needs macOS 26.2+. Boot into Recovery, open Terminal, run `rdma_ctl enable`, reboot.";

const USAGE: &str = "usage: kvrail doctor [--json] [--strict]\n       kvrail --version\n";
const EXIT_USAGE: u8 = 2;

const _: () = assert!(MACOS_VERSION_MIN[0] >= 26);
const _: () = assert!(COMMAND_TIMEOUT.as_millis() == 15_000);
const _: () = assert!(
    COMMAND_POLL_COUNT_MAX as u128 * COMMAND_POLL_INTERVAL.as_millis()
        >= COMMAND_TIMEOUT.as_millis()
);
const _: () = assert!(DETAIL_CHARS_MAX > 0);

/// The outcome of one external command. Callers must tell these cases apart.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CommandResult {
    /// The command exited. `code` is `None` when a signal stopped it.
    Exited {
        code: Option<i32>,
        output: String,
    },
    NotFound,
    Timeout,
    OutputTooLarge,
    /// The command could not start or its output could not be read.
    Failed,
}

impl CommandResult {
    /// Returns the output of a command that exited with status 0.
    fn success_output(&self) -> Option<&str> {
        match self {
            CommandResult::Exited {
                code: Some(0),
                output,
            } => Some(output),
            _ => None,
        }
    }
}

/// Runs a command with a fixed argument vector. Injected so that tests never touch the host.
type Runner<'a> = &'a dyn Fn(&str, &[&str]) -> CommandResult;

type OutputReader = JoinHandle<std::io::Result<Vec<u8>>>;

fn run_command(path: &str, args: &[&str]) -> CommandResult {
    run_command_with_timeout(path, args, COMMAND_TIMEOUT)
}

fn run_command_with_timeout(path: &str, args: &[&str], timeout: Duration) -> CommandResult {
    // Only fixed programs under /usr may run; see the trust boundary in the style profile.
    assert!(path.starts_with("/usr/bin/") || path.starts_with("/usr/sbin/"));
    assert!(timeout <= COMMAND_TIMEOUT);
    // Stderr is dropped: it carries warnings that are not part of the parsed output.
    let spawned = Command::new(path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return CommandResult::NotFound;
        }
        Err(_) => return CommandResult::Failed,
    };
    let Some(stdout) = child.stdout.take() else {
        stop_command(&mut child);
        return CommandResult::Failed;
    };
    // The reader keeps one byte past the limit, so excess is seen without keeping all of it.
    // `Builder::spawn` reports a failure to create the thread, where `thread::spawn` panics.
    let spawned_reader = thread::Builder::new().spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(COMMAND_OUTPUT_BYTES_MAX + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let Ok(reader) = spawned_reader else {
        stop_command(&mut child);
        return CommandResult::Failed;
    };
    wait_command(&mut child, reader, Instant::now() + timeout)
}

fn wait_command(child: &mut Child, reader: OutputReader, deadline: Instant) -> CommandResult {
    let mut reader = Some(reader);
    let mut output: Option<Vec<u8>> = None;
    let mut exit_code: Option<Option<i32>> = None;
    for _ in 0..=COMMAND_POLL_COUNT_MAX {
        if exit_code.is_none() {
            match child.try_wait() {
                Ok(Some(status)) => exit_code = Some(status.code()),
                Ok(None) => {}
                Err(_) => {
                    stop_command(child);
                    return CommandResult::Failed;
                }
            }
        }
        if let Some(handle) = reader.take() {
            if handle.is_finished() {
                let Ok(Ok(bytes)) = handle.join() else {
                    stop_command(child);
                    return CommandResult::Failed;
                };
                output = Some(bytes);
            } else {
                reader = Some(handle);
            }
        }
        if let Some(bytes) = &output {
            if !u64::try_from(bytes.len()).is_ok_and(|length| length <= COMMAND_OUTPUT_BYTES_MAX) {
                stop_command(child);
                return CommandResult::OutputTooLarge;
            }
            if let Some(code) = exit_code {
                let output = String::from_utf8_lossy(bytes).into_owned();
                return CommandResult::Exited { code, output };
            }
        }
        if Instant::now() >= deadline {
            break;
        }
        thread::sleep(COMMAND_POLL_INTERVAL);
    }
    // A reader that is still blocked is left behind: a grandchild can hold the pipe open, and
    // the process exits soon after anyway.
    stop_command(child);
    CommandResult::Timeout
}

fn stop_command(child: &mut Child) {
    // Both calls fail when the child has already exited, which is the state we want anyway.
    let _ = child.kill();
    let _ = child.wait();
}

/// Describes a command outcome that carries no usable output.
fn command_problem(name: &str, result: &CommandResult) -> String {
    match result {
        CommandResult::Exited { code, .. } => exit_code_detail(*code),
        CommandResult::NotFound => format!("{name} not found"),
        CommandResult::Timeout => format!("{name} timed out after 15 s"),
        CommandResult::OutputTooLarge => format!("{name} output is over 1 MiB"),
        CommandResult::Failed => format!("cannot run {name}"),
    }
}

fn exit_code_detail(code: Option<i32>) -> String {
    match code {
        Some(code) => format!("exit code {code}"),
        None => "stopped by a signal".to_string(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Ok,
    Warn,
    Fail,
    Skip,
    Info,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Warn => "warn",
            Status::Fail => "fail",
            Status::Skip => "skip",
            Status::Info => "info",
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Status::Ok => "[ ok ]",
            Status::Warn => "[warn]",
            Status::Fail => "[FAIL]",
            Status::Skip => "[skip]",
            Status::Info => "[info]",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Check {
    id: &'static str,
    title: &'static str,
    status: Status,
    detail: String,
    hint: &'static str,
}

impl Check {
    fn new(
        id: &'static str,
        title: &'static str,
        status: Status,
        detail: &str,
        hint: &'static str,
    ) -> Check {
        assert!(!id.is_empty());
        assert!(!title.is_empty());
        // A hint tells the operator what to fix, so it only exists when something is wrong.
        assert!(hint.is_empty() || matches!(status, Status::Warn | Status::Fail));
        Check {
            id,
            title,
            status,
            detail: detail.to_string(),
            hint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ThunderboltPort {
    receptacle: String,
    speed: String,
    connected: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NetworkService {
    hardware_port: String,
    device: String,
    enabled: bool,
}

fn run_checks(runner: Runner, os: &str, arch: &str) -> Vec<Check> {
    if let Some(check) = check_platform(os, arch) {
        // The other checks call macOS-only commands and would only add noise.
        return vec![check];
    }
    let mut checks = vec![check_macos(runner)];
    checks.extend(check_hardware(runner));
    checks.extend(check_rdma(runner));
    checks.push(check_thunderbolt(runner));
    checks.push(check_bridge(runner));
    checks
}

fn exit_code(checks: &[Check], strict: bool) -> u8 {
    let failed = checks
        .iter()
        // Strict mode is a readiness gate: a check that could not run proves nothing.
        .any(|check| match check.status {
            Status::Fail => true,
            Status::Warn | Status::Skip => strict,
            Status::Ok | Status::Info => false,
        });
    u8::from(failed)
}

fn check_platform(os: &str, arch: &str) -> Option<Check> {
    let title = "macOS on Apple Silicon";
    if os != "macos" {
        let hint = "KVRail targets macOS 26.2+ on Apple Silicon.";
        return Some(Check::new("platform", title, Status::Fail, os, hint));
    }
    if arch != "aarch64" {
        let hint = "Intel Macs are not supported. Under Rosetta, use an aarch64 build of kvrail.";
        return Some(Check::new("platform", title, Status::Fail, arch, hint));
    }
    None
}

fn check_macos(runner: Runner) -> Check {
    let result = runner(SW_VERS, &["-productVersion"]);
    let Some(output) = result.success_output() else {
        let detail = command_problem("sw_vers", &result);
        return Check::new("macos", "macOS version", Status::Warn, &detail, "");
    };
    let Some(version) = parse_macos_version(output) else {
        return Check::new(
            "macos",
            "macOS version",
            Status::Warn,
            "cannot read sw_vers",
            "",
        );
    };
    let title = "macOS >= 26.2 (RDMA over Thunderbolt)";
    if [version[0], version[1]] >= MACOS_VERSION_MIN {
        return Check::new("macos", title, Status::Ok, output.trim(), "");
    }
    let hint = "Update to macOS 26.2 or newer on every node.";
    Check::new("macos", title, Status::Fail, output.trim(), hint)
}

/// Parses exactly `\d{1,4}(\.\d{1,4}){0,2}` into `[major, minor, patch]`.
fn parse_macos_version(text: &str) -> Option<[u32; 3]> {
    let mut version = [0_u32; 3];
    let mut count: usize = 0;
    for component in text.trim().split('.') {
        if count == version.len() {
            return None;
        }
        if component.is_empty()
            || component.len() > VERSION_COMPONENT_DIGITS_MAX
            || !component.bytes().all(|byte| byte.is_ascii_digit())
        {
            return None;
        }
        version[count] = component.parse().ok()?;
        count += 1;
    }
    assert!(count >= 1 && count <= version.len());
    Some(version)
}

/// Parses a plain decimal of at most 20 digits. `str::parse` alone would accept a leading `+`.
fn parse_decimal_u64(text: &str) -> Option<u64> {
    let text = text.trim();
    if text.is_empty()
        || text.len() > DECIMAL_DIGITS_MAX
        || !text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    text.parse().ok()
}

fn check_hardware(runner: Runner) -> Vec<Check> {
    let mut checks = Vec::new();
    let chip = runner(SYSCTL, &["-n", "machdep.cpu.brand_string"]);
    if let Some(chip) = chip
        .success_output()
        .map(str::trim)
        .filter(|chip| !chip.is_empty())
    {
        checks.push(Check::new("chip", "Chip", Status::Info, chip, ""));
    }
    let memory = runner(SYSCTL, &["-n", "hw.memsize"]);
    if let Some(memory_bytes) = memory.success_output().and_then(parse_decimal_u64) {
        let detail = format!("{} GiB", memory_bytes / (1024 * 1024 * 1024));
        checks.push(Check::new(
            "memory",
            "Unified memory",
            Status::Info,
            &detail,
            "",
        ));
    }
    let wired = runner(SYSCTL, &["-n", "iogpu.wired_limit_mb"]);
    if let Some(wired_limit_mib) = wired.success_output().and_then(parse_decimal_u64) {
        // Zero means macOS picks the limit. Raise it with `sudo sysctl iogpu.wired_limit_mb=N`.
        let detail = if wired_limit_mib == 0 {
            "system default".to_string()
        } else {
            format!("{wired_limit_mib} MiB")
        };
        checks.push(Check::new(
            "wired",
            "GPU wired memory limit",
            Status::Info,
            &detail,
            "",
        ));
    }
    checks
}

fn check_rdma(runner: Runner) -> Vec<Check> {
    let title = "RDMA enabled";
    let result = runner(RDMA_CTL, &["status"]);
    let (code, output) = match &result {
        CommandResult::Exited { code, output } => (*code, output.as_str()),
        CommandResult::NotFound => {
            let detail = "rdma_ctl not found";
            return vec![Check::new(
                "rdma_ctl",
                title,
                Status::Fail,
                detail,
                HINT_RDMA_NOT_FOUND,
            )];
        }
        _ => {
            // RDMA state is unknown, and an unknown state must never pass as ready.
            let detail = command_problem("rdma_ctl", &result);
            return vec![Check::new(
                "rdma_ctl",
                title,
                Status::Fail,
                &detail,
                HINT_RDMA_ENABLE,
            )];
        }
    };
    match parse_rdma_status(output) {
        Some(true) => {}
        Some(false) => {
            return vec![Check::new(
                "rdma_ctl",
                title,
                Status::Fail,
                "disabled",
                HINT_RDMA_ENABLE,
            )];
        }
        None => {
            let detail = match output.trim() {
                "" => exit_code_detail(code),
                cleaned => cleaned.to_string(),
            };
            return vec![Check::new(
                "rdma_ctl",
                title,
                Status::Fail,
                &detail,
                HINT_RDMA_ENABLE,
            )];
        }
    }
    vec![
        Check::new("rdma_ctl", title, Status::Ok, "enabled", ""),
        check_ibv(runner),
    ]
}

/// Returns `Some(true)` for "enabled", `Some(false)` for "disabled", `None` for anything else.
fn parse_rdma_status(output: &str) -> Option<bool> {
    // An exact match keeps "disabled" or "not enabled" from being read as enabled.
    match output.trim().to_lowercase().as_str() {
        "enabled" => Some(true),
        "disabled" => Some(false),
        _ => None,
    }
}

fn check_ibv(runner: Runner) -> Check {
    let title = "RDMA verbs devices";
    let result = runner(IBV_DEVICES, &[]);
    // Rows printed by a run that then failed are not evidence of working devices.
    let output = match &result {
        CommandResult::Exited {
            code: Some(0),
            output,
        } => output,
        CommandResult::NotFound => {
            return Check::new("ibv", title, Status::Warn, "ibv_devices not found", "");
        }
        _ => {
            let detail = command_problem("ibv_devices", &result);
            return Check::new("ibv", title, Status::Warn, &detail, "");
        }
    };
    let Some(devices) = parse_ibv_devices(output) else {
        let detail = "cannot parse ibv_devices output";
        return Check::new("ibv", title, Status::Warn, detail, "");
    };
    if devices.is_empty() {
        let hint = "Connect a Thunderbolt 5 cable to a peer Mac (on Mac Studio avoid the port \
                    next to Ethernet), re-plug, and reboot if no rdma_enX appears.";
        return Check::new("ibv", title, Status::Fail, "none listed", hint);
    }
    Check::new("ibv", title, Status::Ok, &devices.join(", "), "")
}

/// Returns the device names, or `None` when there are more than the limit.
fn parse_ibv_devices(output: &str) -> Option<Vec<String>> {
    let mut devices = Vec::new();
    for line in output.lines() {
        let Some(token) = line.split_whitespace().next() else {
            continue;
        };
        if is_rdma_device_name(token) {
            devices.push(token.to_string());
            if devices.len() > RDMA_DEVICE_COUNT_MAX {
                return None;
            }
        }
    }
    Some(devices)
}

/// Matches exactly `rdma_en[0-9]{1,4}`.
fn is_rdma_device_name(token: &str) -> bool {
    let Some(number) = token.strip_prefix("rdma_en") else {
        return false;
    };
    !number.is_empty()
        && number.len() <= RDMA_DEVICE_DIGITS_MAX
        && number.bytes().all(|byte| byte.is_ascii_digit())
}

fn check_thunderbolt(runner: Runner) -> Check {
    let title = "Thunderbolt 5 link (80/120 Gb/s)";
    // Text output is parsed because the crate has no JSON parser and no dependencies.
    let result = runner(SYSTEM_PROFILER, &["SPThunderboltDataType"]);
    let Some(output) = result.success_output() else {
        let detail = command_problem("system_profiler", &result);
        return Check::new("tb", title, Status::Skip, &detail, "");
    };
    if output.trim().is_empty() {
        let detail = "system_profiler printed nothing";
        return Check::new("tb", title, Status::Skip, detail, "");
    }
    let Some(ports) = parse_thunderbolt_ports(output) else {
        let detail = "cannot parse system_profiler output";
        return Check::new("tb", title, Status::Warn, detail, "");
    };
    if ports.is_empty() {
        return Check::new(
            "tb",
            title,
            Status::Warn,
            "no Thunderbolt ports reported",
            "",
        );
    }
    let links: Vec<&ThunderboltPort> = ports.iter().filter(|port| port.connected).collect();
    let detail = thunderbolt_detail(ports.len(), &links);
    if links.is_empty() {
        let hint = "Connect a Thunderbolt 5 cable to the peer Mac.";
        return Check::new("tb", title, Status::Warn, &detail, hint);
    }
    // An idle port shows its capability ("Up to 120 Gb/s"); only a connected port shows a link.
    let fast = links.iter().any(|link| {
        parse_speed_gbps(&link.speed).is_some_and(|speed| speed >= THUNDERBOLT_5_SPEED_GBPS_MIN)
    });
    if fast {
        return Check::new("tb", title, Status::Ok, &detail, "");
    }
    let hint =
        "Link is below 80 Gb/s: use a Thunderbolt 5 cable and Thunderbolt 5 ports on both Macs.";
    Check::new("tb", title, Status::Warn, &detail, hint)
}

fn thunderbolt_detail(port_count: usize, links: &[&ThunderboltPort]) -> String {
    assert!(links.len() <= port_count);
    if links.is_empty() {
        return format!("{port_count} ports, no peer connected");
    }
    let connected: Vec<String> = links
        .iter()
        .map(|link| format!("receptacle {} at {}", link.receptacle, link.speed))
        .collect();
    format!("{port_count} ports, connected: {}", connected.join(", "))
}

fn is_thunderbolt_bus_header(line: &str) -> bool {
    line.ends_with(':')
        && (line.starts_with("Thunderbolt/USB4 Bus") || line.starts_with("Thunderbolt Bus"))
}

/// Parses the ports of this Mac from `system_profiler SPThunderboltDataType` text.
///
/// The ports of this Mac are the "Port:" blocks at the indentation of the first "Vendor Name:"
/// line of each bus. A connected peer device is nested deeper and lists its own ports, which
/// must not count as links of this Mac. Returns `None` when there are more ports than the limit.
fn parse_thunderbolt_ports(text: &str) -> Option<Vec<ThunderboltPort>> {
    let mut ports: Vec<ThunderboltPort> = Vec::new();
    let mut bus_indent: Option<usize> = None;
    let mut bus_field_indent: Option<usize> = None;
    let mut port_field_indent: Option<usize> = None;
    let mut port_open = false;
    for line in text.lines() {
        let content = line.trim_start_matches(' ');
        let trimmed = content.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let indent = line.len() - content.len();
        // A peer device is nested deeper, so only a header at the first bus indent is a bus.
        if is_thunderbolt_bus_header(trimmed) && bus_indent.is_none_or(|bus| bus == indent) {
            bus_indent = Some(indent);
            bus_field_indent = None;
            port_open = false;
            continue;
        }
        let Some(bus) = bus_indent else {
            continue;
        };
        if indent <= bus {
            bus_field_indent = None;
            port_open = false;
            continue;
        }
        if bus_field_indent.is_none() && trimmed.starts_with("Vendor Name:") {
            bus_field_indent = Some(indent);
            continue;
        }
        let Some(bus_fields) = bus_field_indent else {
            continue;
        };
        if indent <= bus_fields {
            port_open = indent == bus_fields && trimmed == "Port:";
            if port_open {
                ports.push(ThunderboltPort {
                    receptacle: String::new(),
                    speed: String::new(),
                    connected: false,
                });
                if ports.len() > THUNDERBOLT_PORT_COUNT_MAX {
                    return None;
                }
                port_field_indent = None;
            }
            continue;
        }
        if !port_open {
            continue;
        }
        let port_fields = *port_field_indent.get_or_insert(indent);
        if indent < port_fields {
            // A line between the bus fields and the port fields starts a nested peer device.
            port_open = false;
            continue;
        }
        if indent == port_fields {
            let port = ports.last_mut();
            assert!(port.is_some());
            if let Some(port) = port {
                read_thunderbolt_port_field(port, trimmed);
            }
        }
    }
    assert!(ports.len() <= THUNDERBOLT_PORT_COUNT_MAX);
    Some(ports)
}

fn read_thunderbolt_port_field(port: &mut ThunderboltPort, line: &str) {
    let Some((key, value)) = line.split_once(':') else {
        return;
    };
    let value = value.trim();
    match key {
        "Status" => port.connected = value == "Device connected",
        "Speed" => port.speed = value.to_string(),
        "Receptacle" => port.receptacle = value.to_string(),
        _ => {}
    }
}

/// Returns the first number in a speed such as "80 Gb/s" or "Up to 120 Gb/s".
/// Reads the first number followed by the unit "Gb/s". "800 Mb/s" is not 800 Gb/s, so a number
/// with any other unit is rejected.
fn parse_speed_gbps(speed: &str) -> Option<u32> {
    let start = speed.find(|c: char| c.is_ascii_digit())?;
    let rest = &speed[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let digits = &rest[..end];
    if digits.len() > VERSION_COMPONENT_DIGITS_MAX {
        return None;
    }
    let unit = rest[end..].strip_prefix(' ').unwrap_or(&rest[end..]);
    if !unit.starts_with("Gb/s") {
        return None;
    }
    digits.parse().ok()
}

fn check_bridge(runner: Runner) -> Check {
    let title = "Thunderbolt Bridge service";
    let result = runner(NETWORKSETUP, &["-listnetworkserviceorder"]);
    let Some(output) = result.success_output() else {
        let detail = command_problem("networksetup", &result);
        return Check::new("bridge", title, Status::Skip, &detail, "");
    };
    let Some(services) = parse_network_services(output) else {
        let detail = "cannot parse networksetup output";
        return Check::new("bridge", title, Status::Skip, detail, "");
    };
    let bridges: Vec<&NetworkService> = services
        .iter()
        .filter(|service| {
            service.hardware_port == THUNDERBOLT_BRIDGE_HARDWARE_PORT
                || service.device == THUNDERBOLT_BRIDGE_DEVICE
        })
        .collect();
    if bridges.iter().any(|bridge| bridge.enabled) {
        let hint = "With 2+ cables between the same Macs it can form a network loop. \
                    Disable it in System Settings -> Network.";
        return Check::new("bridge", title, Status::Warn, "enabled", hint);
    }
    if bridges.is_empty() {
        return Check::new("bridge", title, Status::Ok, "not present", "");
    }
    Check::new("bridge", title, Status::Ok, "disabled", "")
}

/// Parses `networksetup -listnetworkserviceorder`. Returns `None` for an unknown header, a
/// service line without its hardware line, or too many services: the bridge state is then
/// unknown, which must not be reported as "not present".
fn parse_network_services(output: &str) -> Option<Vec<NetworkService>> {
    let mut lines = output.lines().map(str::trim);
    if lines.next() != Some(SERVICE_ORDER_HEADER) {
        return None;
    }
    let mut services = Vec::new();
    let mut pending_enabled: Option<bool> = None;
    for line in lines {
        if let Some(enabled) = pending_enabled.take() {
            let (hardware_port, device) = parse_hardware_line(line)?;
            services.push(NetworkService {
                hardware_port: hardware_port.to_string(),
                device: device.to_string(),
                enabled,
            });
            if services.len() > NETWORK_SERVICE_COUNT_MAX {
                return None;
            }
            continue;
        }
        pending_enabled = parse_service_line(line);
    }
    if pending_enabled.is_some() {
        return None;
    }
    Some(services)
}

/// Reads "(1) Name" as enabled and "(*) Name" as disabled.
fn parse_service_line(line: &str) -> Option<bool> {
    let (marker, _name) = line.strip_prefix('(')?.split_once(") ")?;
    if marker == "*" {
        return Some(false);
    }
    let numbered = !marker.is_empty()
        && marker.len() <= VERSION_COMPONENT_DIGITS_MAX
        && marker.bytes().all(|byte| byte.is_ascii_digit());
    numbered.then_some(true)
}

/// Reads "(Hardware Port: X, Device: Y)" into `(X, Y)`.
fn parse_hardware_line(line: &str) -> Option<(&str, &str)> {
    line.strip_prefix("(Hardware Port: ")?
        .strip_suffix(')')?
        .rsplit_once(", Device: ")
}

/// Makes command text safe for a terminal and cuts it to at most `DETAIL_CHARS_MAX` chars.
fn clean_detail(detail: &str) -> String {
    let clean = |c: char| if c.is_control() { '?' } else { c };
    if detail.chars().nth(DETAIL_CHARS_MAX).is_none() {
        return detail.chars().map(clean).collect();
    }
    let mut cleaned: String = detail
        .chars()
        .take(DETAIL_CHARS_MAX - 3)
        .map(clean)
        .collect();
    cleaned.push_str("...");
    assert_eq!(cleaned.chars().count(), DETAIL_CHARS_MAX);
    cleaned
}

fn render_text(checks: &[Check]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for check in checks {
        let detail = clean_detail(&check.detail);
        if detail.is_empty() {
            lines.push(format!("{} {}", check.status.tag(), check.title));
        } else {
            lines.push(format!(
                "{} {}: {}",
                check.status.tag(),
                check.title,
                detail
            ));
        }
        if !check.hint.is_empty() {
            lines.push(format!("       -> {}", check.hint));
        }
    }
    lines.join("\n")
}

/// Renders the same text as Python `json.dumps([asdict(check) ...], indent=2)`.
/// The detail is cleaned as in the text report, so both outputs carry the same value.
fn render_json(checks: &[Check]) -> String {
    if checks.is_empty() {
        return "[]".to_string();
    }
    let mut json = String::from("[\n");
    for (check_index, check) in checks.iter().enumerate() {
        if check_index > 0 {
            json.push_str(",\n");
        }
        json.push_str("  {\n");
        let detail = clean_detail(&check.detail);
        let fields = [
            ("id", check.id),
            ("title", check.title),
            ("status", check.status.name()),
            ("detail", detail.as_str()),
            ("hint", check.hint),
        ];
        for (field_index, (name, value)) in fields.iter().enumerate() {
            json.push_str("    ");
            push_json_string(&mut json, name);
            json.push_str(": ");
            push_json_string(&mut json, value);
            json.push_str(if field_index + 1 < fields.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        json.push_str("  }");
    }
    json.push_str("\n]");
    json
}

/// Appends a JSON string literal. Like Python `ensure_ascii`, it escapes everything outside
/// printable ASCII, so the output is safe for any terminal and any consumer encoding.
fn push_json_string(json: &mut String, value: &str) {
    json.push('"');
    for c in value.chars() {
        match c {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            '\u{8}' => json.push_str("\\b"),
            '\u{c}' => json.push_str("\\f"),
            ' '..='~' => json.push(c),
            _ => {
                let mut units = [0_u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    // Writing to a `String` cannot fail.
                    let _ = write!(json, "\\u{unit:04x}");
                }
            }
        }
    }
    json.push('"');
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cli {
    Help,
    Version,
    Doctor { json: bool, strict: bool },
}

/// Returns `None` for a usage error: no command or an unknown word. A repeated flag is
/// accepted, as Python argparse accepts it.
fn parse_args(args: &[OsString]) -> Option<Cli> {
    let mut words: Vec<&str> = Vec::new();
    for arg in args {
        words.push(arg.to_str()?);
    }
    match words.as_slice() {
        ["--help" | "-h"] => Some(Cli::Help),
        ["--version"] => Some(Cli::Version),
        ["doctor", flags @ ..] => {
            let mut json = false;
            let mut strict = false;
            for flag in flags {
                match *flag {
                    "--json" => json = true,
                    "--strict" => strict = true,
                    "--help" | "-h" => return Some(Cli::Help),
                    _ => return None,
                }
            }
            Some(Cli::Doctor { json, strict })
        }
        _ => None,
    }
}

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let Some(cli) = parse_args(&args) else {
        // A failed write to stderr cannot be reported anywhere else.
        let _ = std::io::stderr().write_all(USAGE.as_bytes());
        return ExitCode::from(EXIT_USAGE);
    };
    let (report, code) = match cli {
        // Help that was asked for is not an error, so it goes to stdout with exit 0.
        Cli::Help => (USAGE.trim_end().to_string(), 0),
        Cli::Version => (format!("kvrail {}", env!("CARGO_PKG_VERSION")), 0),
        Cli::Doctor { json, strict } => {
            let checks = run_checks(&run_command, std::env::consts::OS, std::env::consts::ARCH);
            let report = if json {
                render_json(&checks)
            } else {
                render_text(&checks)
            };
            (report, exit_code(&checks, strict))
        }
    };
    // `println!` would panic on a closed pipe; a report that cannot be written is a failure.
    if writeln!(std::io::stdout(), "{report}").is_err() {
        return ExitCode::from(1);
    }
    ExitCode::from(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TB_NO_PEER: &str = include_str!("fixtures/thunderbolt_mac_studio_no_peer.txt");
    const TB_PEER_80: &str = include_str!("fixtures/thunderbolt_mac_studio_peer_80.txt");
    const IBV_EMPTY: &str = include_str!("fixtures/ibv_devices_empty.txt");
    const SERVICES_BRIDGE_ENABLED: &str = include_str!("fixtures/service_order_bridge_enabled.txt");

    fn exited(output: &str) -> CommandResult {
        CommandResult::Exited {
            code: Some(0),
            output: output.to_string(),
        }
    }

    /// Answers each command line from a table; any other command is not found.
    fn fake_runner(
        responses: Vec<(&'static str, CommandResult)>,
    ) -> impl Fn(&str, &[&str]) -> CommandResult {
        move |path: &str, args: &[&str]| {
            let mut command_line = path.to_string();
            for arg in args {
                command_line.push(' ');
                command_line.push_str(arg);
            }
            responses
                .iter()
                .find(|(line, _)| *line == command_line)
                .map_or(CommandResult::NotFound, |(_, result)| result.clone())
        }
    }

    fn thunderbolt(output: &str) -> Check {
        let runner = fake_runner(vec![("/usr/sbin/system_profiler SPThunderboltDataType", {
            exited(output)
        })]);
        check_thunderbolt(&runner)
    }

    fn bridge(output: &str) -> Check {
        let runner = fake_runner(vec![(
            "/usr/sbin/networksetup -listnetworkserviceorder",
            exited(output),
        )]);
        check_bridge(&runner)
    }

    #[test]
    fn ibv_failed_run_is_not_evidence() {
        // Goal: device rows from an ibv_devices run that exits nonzero do not pass.
        let failed = CommandResult::Exited {
            code: Some(1),
            output: "    rdma_en2  0000\n".to_string(),
        };
        let checks = rdma(exited("enabled\n"), failed);
        assert_eq!(
            (checks[1].status, checks[1].detail.as_str()),
            (Status::Warn, "exit code 1")
        );
    }

    fn rdma(status: CommandResult, ibv: CommandResult) -> Vec<Check> {
        let runner = fake_runner(vec![
            ("/usr/bin/rdma_ctl status", status),
            ("/usr/bin/ibv_devices", ibv),
        ]);
        check_rdma(&runner)
    }

    fn args(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    #[test]
    fn thunderbolt_idle_ports_warn() {
        // Goal: idle ports say "Up to 120 Gb/s"; that is a capability, so it must not pass.
        let check = thunderbolt(TB_NO_PEER);
        assert_eq!(check.status, Status::Warn);
        assert_eq!(check.detail, "6 ports, no peer connected");
        assert_eq!(check.hint, "Connect a Thunderbolt 5 cable to the peer Mac.");
    }

    #[test]
    fn thunderbolt_connected_80_ok() {
        // Goal: a bus-level port with "Device connected" at 80 Gb/s is a Thunderbolt 5 link.
        let check = thunderbolt(TB_PEER_80);
        assert_eq!(check.status, Status::Ok);
        assert_eq!(check.detail, "6 ports, connected: receptacle 4 at 80 Gb/s");
    }

    #[test]
    fn thunderbolt_connected_40_warns() {
        // Goal: a connected link below 80 Gb/s warns, even though the peer upstream says 80.
        let fixture = TB_PEER_80.replacen("Speed: 80 Gb/s", "Speed: 40 Gb/s", 1);
        let check = thunderbolt(&fixture);
        assert_eq!(check.status, Status::Warn);
        assert_eq!(check.detail, "6 ports, connected: receptacle 4 at 40 Gb/s");
        assert!(check.hint.starts_with("Link is below 80 Gb/s"));
    }

    #[test]
    fn thunderbolt_nested_peer_ports_ignored() {
        // Goal: connected 80 Gb/s ports of a nested peer device are not links of this Mac.
        let fixture =
            TB_PEER_80.replacen("Status: Device connected", "Status: No device connected", 1);
        let ports = parse_thunderbolt_ports(&fixture).unwrap_or_default();
        assert_eq!(ports.len(), 6);
        assert!(ports.iter().all(|port| !port.connected));
        let check = thunderbolt(&fixture);
        assert_eq!(check.status, Status::Warn);
        assert_eq!(check.detail, "6 ports, no peer connected");
    }

    #[test]
    fn thunderbolt_nested_fields_ignored() {
        // Goal: none of these adds a link to this Mac: a nested device whose fields line up with
        // the port fields, lines deeper than the port fields, or a nested device named like a bus.
        let idle = "  Thunderbolt/USB4 Bus 0:\n    Vendor Name: Apple Inc.\n    Port:\n      \
                    Status: No device connected\n      Speed: Up to 120 Gb/s\n";
        let peer_aligned = format!("{idle}     Peer:\n      Status: Device connected\n");
        let deeper = format!("{idle}        Status: Device connected\n");
        let bus_like_peer = format!(
            "{idle}      Thunderbolt Bus 9:\n        Vendor Name: Peer\n        Port:\n          \
             Status: Device connected\n"
        );
        for fixture in [peer_aligned, deeper, bus_like_peer] {
            let ports = parse_thunderbolt_ports(&fixture).unwrap_or_default();
            assert_eq!(ports.len(), 1);
            assert!(!ports[0].connected, "{fixture}");
        }
    }

    #[test]
    fn thunderbolt_ports_read_fields() {
        // Goal: the parser reads status, speed and receptacle of each bus-level port in order.
        let ports = parse_thunderbolt_ports(TB_PEER_80).unwrap_or_default();
        let receptacles: Vec<&str> = ports.iter().map(|port| port.receptacle.as_str()).collect();
        assert_eq!(receptacles, ["6", "5", "4", "3", "2", "1"]);
        assert_eq!(
            ports[2],
            ThunderboltPort {
                receptacle: "4".to_string(),
                speed: "80 Gb/s".to_string(),
                connected: true,
            }
        );
        assert_eq!(ports[0].speed, "Up to 120 Gb/s");
    }

    #[test]
    fn thunderbolt_port_limit() {
        // Goal: more ports than the limit is a parse failure, not an unbounded list.
        let block = "  Thunderbolt/USB4 Bus 0:\n    Vendor Name: Apple Inc.\n    Port:\n      \
                     Status: Device connected\n      Speed: 80 Gb/s\n";
        assert_eq!(
            parse_thunderbolt_ports(&block.repeat(64)).map(|ports| ports.len()),
            Some(64)
        );
        assert_eq!(parse_thunderbolt_ports(&block.repeat(65)), None);
        let check = thunderbolt(&block.repeat(65));
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Warn, "cannot parse system_profiler output")
        );
    }

    #[test]
    fn thunderbolt_unavailable_or_empty() {
        // Goal: a failed command skips with its reason, and output without ports warns.
        let runner = fake_runner(vec![]);
        let check = check_thunderbolt(&runner);
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Skip, "system_profiler not found")
        );
        let runner = fake_runner(vec![(
            "/usr/sbin/system_profiler SPThunderboltDataType",
            CommandResult::Timeout,
        )]);
        assert_eq!(
            check_thunderbolt(&runner).detail,
            "system_profiler timed out after 15 s"
        );
        let check = thunderbolt("Thunderbolt/USB4:\n\n");
        assert_eq!(check.status, Status::Warn);
        assert_eq!(check.detail, "no Thunderbolt ports reported");
    }

    #[test]
    fn speed_parse() {
        // Goal: the first number of a speed is read, and absurd numbers are rejected.
        assert_eq!(parse_speed_gbps("80 Gb/s"), Some(80));
        assert_eq!(parse_speed_gbps("Up to 120 Gb/s"), Some(120));
        assert_eq!(parse_speed_gbps("Up to 40 Gb/s"), Some(40));
        assert_eq!(parse_speed_gbps("fast"), None);
        assert_eq!(parse_speed_gbps("800 Mb/s"), None);
        assert_eq!(parse_speed_gbps("80Gb/s"), Some(80));
        assert_eq!(parse_speed_gbps("99999 Gb/s"), None);
    }

    #[test]
    fn bridge_enabled_warns() {
        // Goal: the real service order of this Mac, with the bridge enabled, warns.
        let check = bridge(SERVICES_BRIDGE_ENABLED);
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Warn, "enabled")
        );
        assert!(check.hint.contains("network loop"));
    }

    #[test]
    fn bridge_renamed_still_detected() {
        // Goal: a renamed bridge service is found by its hardware port or its device.
        let renamed = SERVICES_BRIDGE_ENABLED.replace("(3) Thunderbolt Bridge", "(3) My Link");
        assert_eq!(bridge(&renamed).status, Status::Warn);
        let device_only = services("(1) Link\n(Hardware Port: Something, Device: bridge0)\n");
        assert_eq!(bridge(&device_only).status, Status::Warn);
    }

    #[test]
    fn bridge_disabled_or_absent_ok() {
        // Goal: a disabled bridge and a missing bridge are both safe.
        let disabled = SERVICES_BRIDGE_ENABLED.replace("(3) Thunderbolt Bridge", "(*) TB");
        let check = bridge(&disabled);
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Ok, "disabled")
        );
        let absent = services("(1) Ethernet\n(Hardware Port: Ethernet, Device: en0)\n");
        let check = bridge(&absent);
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Ok, "not present")
        );
        let named_only =
            services("(1) Thunderbolt Bridge\n(Hardware Port: Ethernet, Device: en5)\n");
        assert_eq!(bridge(&named_only).detail, "not present");
    }

    #[test]
    fn bridge_unknown_output_is_not_absent() {
        // Goal: output that cannot be read leaves the bridge state unknown (skip), never
        // "not present": no header, an empty list, or a service without its hardware line.
        let incomplete = SERVICES_BRIDGE_ENABLED
            .replace("(Hardware Port: Thunderbolt Bridge, Device: bridge0)\n", "");
        let headless = SERVICES_BRIDGE_ENABLED
            .split_once('\n')
            .map_or("", |(_, rest)| rest);
        for output in ["", "\n", "garbage\n", headless, incomplete.as_str()] {
            let check = bridge(output);
            assert_eq!(
                (check.status, check.detail.as_str()),
                (Status::Skip, "cannot parse networksetup output"),
                "{output:?}"
            );
        }
    }

    fn services(body: &str) -> String {
        format!("{SERVICE_ORDER_HEADER}\n{body}")
    }

    #[test]
    fn bridge_unavailable_or_too_many_services_skip() {
        // Goal: a failed command and a service list over the limit both skip.
        let check = check_bridge(&fake_runner(vec![]));
        assert_eq!(
            (check.status, check.detail.as_str()),
            (Status::Skip, "networksetup not found")
        );
        let service = "(1) Ethernet\n(Hardware Port: Ethernet, Device: en0)\n";
        assert_eq!(
            parse_network_services(&services(&service.repeat(64))).map(|s| s.len()),
            Some(64)
        );
        let check = bridge(&services(&service.repeat(65)));
        assert_eq!(check.status, Status::Skip);
        assert_eq!(check.detail, "cannot parse networksetup output");
    }

    #[test]
    fn rdma_disabled_fails_without_ibv() {
        // Goal: "disabled" is never read as enabled, and the verbs check does not run.
        let checks = rdma(exited("disabled\n"), exited("rdma_en2 0000\n"));
        assert_eq!(checks.len(), 1);
        assert_eq!(
            (checks[0].status, checks[0].detail.as_str()),
            (Status::Fail, "disabled")
        );
        assert_eq!(checks[0].hint, HINT_RDMA_ENABLE);
        assert_eq!(parse_rdma_status("DISABLED\n"), Some(false));
        assert_eq!(parse_rdma_status("not enabled"), None);
        assert_eq!(parse_rdma_status(" Enabled\n"), Some(true));
    }

    #[test]
    fn rdma_not_found_or_unknown_fails() {
        // Goal: a missing rdma_ctl and an unknown answer fail with a useful detail.
        let checks = rdma(CommandResult::NotFound, CommandResult::NotFound);
        assert_eq!(checks[0].detail, "rdma_ctl not found");
        assert_eq!(checks[0].hint, HINT_RDMA_NOT_FOUND);
        let checks = rdma(exited("weird state\n"), CommandResult::NotFound);
        assert_eq!(
            (checks[0].status, checks[0].detail.as_str()),
            (Status::Fail, "weird state")
        );
        let empty = CommandResult::Exited {
            code: Some(3),
            output: String::new(),
        };
        assert_eq!(
            rdma(empty, CommandResult::NotFound)[0].detail,
            "exit code 3"
        );
        let checks = rdma(CommandResult::Timeout, CommandResult::NotFound);
        assert_eq!(checks[0].status, Status::Fail);
    }

    #[test]
    fn ibv_devices_outcomes() {
        // Goal: each ibv_devices outcome maps to its own status after RDMA is enabled.
        let checks = rdma(exited("enabled\n"), exited(IBV_EMPTY));
        assert_eq!(
            (checks[0].status, checks[0].detail.as_str()),
            (Status::Ok, "enabled")
        );
        assert_eq!(
            (checks[1].status, checks[1].detail.as_str()),
            (Status::Fail, "none listed")
        );
        let checks = rdma(exited("enabled"), CommandResult::NotFound);
        assert_eq!(
            (checks[1].status, checks[1].detail.as_str()),
            (Status::Warn, "ibv_devices not found")
        );
        let listed = format!("{IBV_EMPTY}    rdma_en2  \t0000\n    rdma_en3  \t0000\n");
        let checks = rdma(exited("enabled"), exited(&listed));
        assert_eq!(
            (checks[1].status, checks[1].detail.as_str()),
            (Status::Ok, "rdma_en2, rdma_en3")
        );
        let many = "rdma_en1\n".repeat(65);
        assert_eq!(
            rdma(exited("enabled"), exited(&many))[1].status,
            Status::Warn
        );
    }

    #[test]
    fn rdma_device_names() {
        // Goal: only `rdma_en[0-9]{1,4}` names count as devices.
        assert!(is_rdma_device_name("rdma_en0"));
        assert!(is_rdma_device_name("rdma_en1234"));
        assert!(!is_rdma_device_name("rdma_en"));
        assert!(!is_rdma_device_name("rdma_en12345"));
        assert!(!is_rdma_device_name("rdma_enX"));
        assert!(!is_rdma_device_name("xrdma_en1"));
    }

    #[test]
    fn macos_version_parse() {
        // Goal: only the canonical version form parses; everything else is rejected.
        assert_eq!(parse_macos_version("27.2.1\n"), Some([27, 2, 1]));
        assert_eq!(parse_macos_version("26"), Some([26, 0, 0]));
        for text in [
            "", "abc", "26.2.1.5", "99999.1", "26.", ".2", "26.x", "+26.2", "26 .2",
        ] {
            assert_eq!(parse_macos_version(text), None, "{text}");
        }
    }

    #[test]
    fn macos_version_compare() {
        // Goal: 26.1.9 is too old, 26.2 and 27.0 are new enough, unreadable output warns.
        let check_for = |output: &str| {
            check_macos(&fake_runner(vec![(
                "/usr/bin/sw_vers -productVersion",
                exited(output),
            )]))
        };
        let old = check_for("26.1.9\n");
        assert_eq!(
            (old.status, old.hint),
            (Status::Fail, "Update to macOS 26.2 or newer on every node.")
        );
        assert_eq!(check_for("26.2").status, Status::Ok);
        assert_eq!(check_for("27.0\n").detail, "27.0");
        assert_eq!(check_for("27.0\n").status, Status::Ok);
        let garbage = check_for("Version 27");
        assert_eq!(
            (garbage.status, garbage.title),
            (Status::Warn, "macOS version")
        );
        assert_eq!(garbage.detail, "cannot read sw_vers");
        assert_eq!(
            check_macos(&fake_runner(vec![])).detail,
            "sw_vers not found"
        );
    }

    #[test]
    fn hardware_info() {
        // Goal: sysctl values become info checks, and a failed or odd value is skipped silently.
        let runner = fake_runner(vec![
            (
                "/usr/sbin/sysctl -n machdep.cpu.brand_string",
                exited("Apple M3 Ultra\n"),
            ),
            ("/usr/sbin/sysctl -n hw.memsize", exited("549755813887\n")),
            ("/usr/sbin/sysctl -n iogpu.wired_limit_mb", exited("0\n")),
        ]);
        let details: Vec<String> = check_hardware(&runner)
            .into_iter()
            .map(|check| check.detail)
            .collect();
        assert_eq!(details, ["Apple M3 Ultra", "511 GiB", "system default"]);
        let runner = fake_runner(vec![
            (
                "/usr/sbin/sysctl -n hw.memsize",
                exited("123456789012345678901\n"),
            ),
            ("/usr/sbin/sysctl -n iogpu.wired_limit_mb", exited("4096")),
        ]);
        let checks = check_hardware(&runner);
        assert_eq!(checks.len(), 1);
        assert_eq!(
            (checks[0].id, checks[0].detail.as_str()),
            ("wired", "4096 MiB")
        );
        assert_eq!(parse_decimal_u64("+5"), None);
    }

    #[test]
    fn platform_gate() {
        // Goal: a host that is not macOS on Apple Silicon gets one fail and no other checks.
        let checks = run_checks(&fake_runner(vec![]), "linux", "x86_64");
        assert_eq!(checks.len(), 1);
        assert_eq!((checks[0].id, checks[0].status), ("platform", Status::Fail));
        assert_eq!(
            check_platform("macos", "x86_64").map(|check| check.detail),
            Some("x86_64".into())
        );
        assert_eq!(check_platform("macos", "aarch64"), None);
    }

    fn this_mac_runner() -> impl Fn(&str, &[&str]) -> CommandResult {
        fake_runner(vec![
            ("/usr/bin/sw_vers -productVersion", exited("27.2.1\n")),
            (
                "/usr/sbin/sysctl -n machdep.cpu.brand_string",
                exited("Apple M3 Ultra\n"),
            ),
            ("/usr/sbin/sysctl -n hw.memsize", exited("549755813888\n")),
            ("/usr/sbin/sysctl -n iogpu.wired_limit_mb", exited("0\n")),
            ("/usr/bin/rdma_ctl status", exited("disabled\n")),
            (
                "/usr/sbin/system_profiler SPThunderboltDataType",
                exited(TB_NO_PEER),
            ),
            (
                "/usr/sbin/networksetup -listnetworkserviceorder",
                exited(SERVICES_BRIDGE_ENABLED),
            ),
        ])
    }

    #[test]
    fn full_run_on_this_mac() {
        // Goal: the outputs of the reference Mac give the verdict that was checked by hand.
        let checks = run_checks(&this_mac_runner(), "macos", "aarch64");
        let summary: Vec<(&str, Status)> = checks
            .iter()
            .map(|check| (check.id, check.status))
            .collect();
        assert_eq!(
            summary,
            [
                ("macos", Status::Ok),
                ("chip", Status::Info),
                ("memory", Status::Info),
                ("wired", Status::Info),
                ("rdma_ctl", Status::Fail),
                ("tb", Status::Warn),
                ("bridge", Status::Warn),
            ]
        );
        assert_eq!(exit_code(&checks, false), 1);
    }

    #[test]
    fn exit_code_strict() {
        // Goal: warnings and skips fail only with --strict, and fails always fail.
        let warn = Check::new("tb", "Thunderbolt", Status::Warn, "", "");
        let skip = Check::new("bridge", "Bridge", Status::Skip, "", "");
        assert_eq!(exit_code(&[warn.clone(), skip.clone()], false), 0);
        assert_eq!(exit_code(&[warn, skip.clone()], true), 1);
        assert_eq!(exit_code(std::slice::from_ref(&skip), false), 0);
        assert_eq!(exit_code(std::slice::from_ref(&skip), true), 1);
        let fail = Check::new("macos", "macOS", Status::Fail, "", "");
        assert_eq!(exit_code(&[skip, fail], false), 1);
    }

    #[test]
    #[should_panic(expected = "hint.is_empty()")]
    fn hint_on_ok_is_a_programmer_error() {
        // Goal: the constructor rejects a hint on a check that needs no action.
        let _ = Check::new("macos", "macOS", Status::Ok, "27.2.1", "Update.");
    }

    #[test]
    fn text_render() {
        // Goal: tags, details and hints are laid out like the Python renderer.
        let checks = [
            Check::new("macos", "macOS", Status::Ok, "27.2.1", ""),
            Check::new("bridge", "Bridge", Status::Warn, "enabled", "Disable it."),
            Check::new("ibv", "Verbs", Status::Fail, "", ""),
            Check::new("tb", "TB", Status::Skip, "x", ""),
            Check::new("chip", "Chip", Status::Info, "M3", ""),
        ];
        let expected = "[ ok ] macOS: 27.2.1\n[warn] Bridge: enabled\n       -> Disable it.\n\
                        [FAIL] Verbs\n[skip] TB: x\n[info] Chip: M3";
        assert_eq!(render_text(&checks), expected);
    }

    #[test]
    fn text_render_cleans_detail() {
        // Goal: control characters cannot reach the terminal, and long details are cut safely.
        assert_eq!(clean_detail("a\u{1b}[31mb\nc\u{7f}"), "a?[31mb?c?");
        let long = "\u{e9}".repeat(DETAIL_CHARS_MAX + 1);
        let cleaned = clean_detail(&long);
        assert_eq!(cleaned.chars().count(), DETAIL_CHARS_MAX);
        assert!(cleaned.ends_with("\u{e9}..."));
        assert_eq!(
            clean_detail(&"x".repeat(DETAIL_CHARS_MAX)),
            "x".repeat(DETAIL_CHARS_MAX)
        );
    }

    #[test]
    fn json_string_escaping() {
        // Goal: quotes, backslashes and control characters are escaped like Python json.dumps.
        let mut json = String::new();
        push_json_string(&mut json, "a\"b\\c\nd\u{1}e\t\u{7f}\u{e9}\u{1f600}/");
        assert_eq!(json, r#""a\"b\\c\nd\u0001e\t\u007f\u00e9\ud83d\ude00/""#);
    }

    #[test]
    fn json_render_layout() {
        // Goal: the array matches Python json.dumps(indent=2) byte for byte.
        let checks = [
            Check::new("macos", "macOS", Status::Ok, "27.2.1", ""),
            Check::new(
                "bridge",
                "Bridge",
                Status::Warn,
                "enabled",
                "Disable \"it\".",
            ),
        ];
        let expected = r#"[
  {
    "id": "macos",
    "title": "macOS",
    "status": "ok",
    "detail": "27.2.1",
    "hint": ""
  },
  {
    "id": "bridge",
    "title": "Bridge",
    "status": "warn",
    "detail": "enabled",
    "hint": "Disable \"it\"."
  }
]"#;
        assert_eq!(render_json(&checks), expected);
        assert_eq!(render_json(&[]), "[]");
    }

    #[test]
    fn cli_parse() {
        // Goal: only the documented command lines parse; everything else is a usage error.
        assert_eq!(parse_args(&args(&["--version"])), Some(Cli::Version));
        assert_eq!(parse_args(&args(&["doctor", "--help"])), Some(Cli::Help));
        assert_eq!(
            parse_args(&args(&["doctor", "--json", "--help"])),
            Some(Cli::Help)
        );
        assert_eq!(
            parse_args(&args(&["doctor", "--json", "--json"])),
            Some(Cli::Doctor {
                json: true,
                strict: false
            })
        );
        assert_eq!(
            parse_args(&args(&["doctor"])),
            Some(Cli::Doctor {
                json: false,
                strict: false
            })
        );
        assert_eq!(
            parse_args(&args(&["doctor", "--strict", "--json"])),
            Some(Cli::Doctor {
                json: true,
                strict: true
            })
        );
        for words in [
            &[][..],
            &["bogus"][..],
            &["doctor", "--bogus"][..],
            &["--version", "doctor"][..],
            &["--json", "doctor"][..],
        ] {
            assert_eq!(parse_args(&args(words)), None, "{words:?}");
        }
    }

    #[test]
    fn command_runner_outcomes() {
        // Goal: the runner tells exit, missing binary, oversized output and timeout apart.
        assert_eq!(
            run_command("/usr/bin/printf", &["a%sb", "x"]),
            exited("axb")
        );
        let failed = run_command("/usr/bin/false", &[]);
        assert_eq!(
            failed,
            CommandResult::Exited {
                code: Some(1),
                output: String::new()
            }
        );
        assert_eq!(
            run_command("/usr/bin/kvrail-missing", &[]),
            CommandResult::NotFound
        );
        assert_eq!(
            run_command("/usr/bin/yes", &[]),
            CommandResult::OutputTooLarge
        );
        let started = Instant::now();
        let result = run_command_with_timeout(
            "/usr/bin/tail",
            &["-f", "/dev/null"],
            Duration::from_millis(200),
        );
        assert_eq!(result, CommandResult::Timeout);
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}

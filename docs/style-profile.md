# KVRail style profile

KVRail follows TigerStyle as restated in "Adapting TigerStyle to Your Codebase" (revision 5,
2026-11-17). This file is the project profile that the guide asks for: which blocks apply, which
mechanisms we chose, every limit and its reaction, and the exceptions. Where this file is silent,
the guide holds.

## Scope (v0.0.1)

v0.0.1 contains one program, `kvrail doctor`, in two implementations with the same contract:
`src/kvrail` (Python, stdlib only) and `rust/` (Rust, zero dependencies).

Blocks in force:

- Baseline: Limits and errors, Security (threat model, trust boundaries, limits as defense,
  supply chain, verification).
- Profile: Control flow, Assertions, State and numbers, Dependencies and tooling.
- Conventions: Functions, Naming, Comments, commits and layout.

Blocks out of scope, because their subject does not exist in v0.0.1: Secrets (the program holds
no secrets), Cryptography, Concurrency (the program is single-threaded and runs commands one at a
time), Performance (no throughput or latency requirement; "Let no showstopper through" still
holds). The data plane (wire protocol, slab store) will reopen Performance, Concurrency and the
memory rules.

## Threat model

- What we protect: the correctness of the verdict (a "ok" must never be shown for a node that is
  not ready), and the host (the tool must not hang, run arbitrary programs or corrupt a terminal).
- Inputs: command-line arguments, and the output of six macOS system commands. Their output is
  parsed as untrusted text: it changes between macOS releases and can be large or malformed.
- Trust boundary: every command runs from an absolute path under `/usr/bin` or `/usr/sbin` with a
  fixed argument vector, so `PATH` cannot redirect it and no input reaches a command line.
- Out of scope: an attacker who can replace files under `/usr` already controls the machine.
- The program opens no network connections, writes no files and needs no privileges.

## Mechanisms

- Assertions: Python `assert` is removed by `python -O`, so Python code uses
  `kvrail.doctor.invariant()`, which always raises `AssertionError`. Rust uses `assert!`, never
  `debug_assert!`. A failed assertion stops the process; nothing catches it.
- Errors: a missing command, a timeout, oversized output or an unknown format is an operating
  error. It becomes a `skip`, `warn` or `fail` check, never an exception or a panic.
- Recursion: none in our code. Python's `json.loads` recurses internally; its input is capped at
  1 MiB and a `RecursionError` is handled as a parse failure.
- Numbers: Python integers are parsed only from at most 20 digits, so no value can grow without
  bound. Rust uses `u32`/`u64`, never `usize` in data that crosses the boundary.
- Output: text that came from a command is cleaned of control characters before it reaches the
  terminal. JSON output is produced by an encoder, never by string joining.

## Limits

| Limit                     | Value     | Scope       | Checked                  | On excess                    |
|---------------------------|-----------|-------------|--------------------------|------------------------------|
| Command run time          | 15 s      | Per command | While reading output     | Kill it; check skip or fail  |
| Command output            | 1 MiB     | Per command | Before the bytes are kept| Kill it; check skip or fail  |
| Thunderbolt ports         | 64        | Per run     | In the parser            | Parse failure, check warn    |
| Network services          | 64        | Per run     | In the parser            | Parse failure, check skip    |
| RDMA devices              | 64        | Per run     | In the parser            | Parse failure, check warn    |
| Version component         | 4 digits  | Per value   | In the parser            | Parse failure, check warn    |
| Check detail shown        | 200 chars | Per check   | At render                | Truncate with "..."          |
| Function length           | 70 lines  | Per function| Test (Python), clippy    | Fail the build               |
| Line length               | 100 cols  | Per line    | Test (Python), rustfmt   | Fail the build               |

Six commands at 15 s each bound one run at 90 s.

## Dependencies and tooling

- Runtime dependencies: none, in both implementations.
- Build: `hatchling` (Python wheel), Cargo. `Cargo.lock` is committed.
- Minimum Rust version: 1.82, the first release with `Option::is_none_or`. CI proves it.
- CI actions are pinned by commit hash. A `v<version>` tag runs `release.yml`, which publishes
  both packages through OIDC trusted publishing; no long-lived
  token is stored in the repository.
- Checks that run in CI: `pytest` (includes the line and function length test), `cargo fmt
  --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.

## Exceptions

None.

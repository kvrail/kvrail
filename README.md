# KVRail

RDMA-over-Thunderbolt KV-cache store and context coprocessor for Apple Silicon
(Mac mini + Mac Studio, macOS 26.2+). Early R&D; see https://kvrail.org.

v0.0.1 ships one tool, `kvrail doctor`. It checks whether a Mac is ready for RDMA over
Thunderbolt 5: macOS version, chip, memory, GPU wired limit, `rdma_ctl`, `ibv_devices`,
a connected Thunderbolt 5 link at 80/120 Gb/s, and the Thunderbolt Bridge loop risk.

    pip install kvrail          # or: cargo install kvrail
    kvrail doctor
    kvrail doctor --json --strict

Exit codes: 0 when every check passes, 1 when a check fails, 2 on a usage error. With
`--strict`, a warning or a skipped check (a command that could not run) also exits 1.

`kvrail doctor` runs only fixed system commands from `/usr/bin` and `/usr/sbin`, with a 15 s
timeout and a 1 MiB output limit each. It needs no privileges and opens no network connections.

## Layout

- `src/kvrail`, `tests/`, `fixtures/`: Python package (stdlib only)
- `rust/`: Rust crate with the same `doctor` command, zero dependencies
- `docs/style-profile.md`: the TigerStyle project profile, limits and threat model
- `.github/workflows/`: CI and trusted-publishing workflows

## License

Apache-2.0, see `LICENSE`.

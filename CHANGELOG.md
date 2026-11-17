# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `docs/architecture.md`: architecture specification with use cases, hardware and cabling rules,
  transport and storage design, limits and the M0-M4 milestones.
- Each `v<version>` tag also creates a GitHub Release with the wheel, the sdist and the crate,
  taken from PyPI and crates.io, and notes from this file.
- README badges for PyPI, crates.io, CI and the license.

### Changed

- The TigerStyle guide now lives in [kvrail/tigerstyle](https://github.com/kvrail/tigerstyle);
  `docs/style-profile.md` links to its pinned revision r5.

## [0.0.1] - 2026-11-17

### Added

- `kvrail doctor [--json] [--strict]` in Python and Rust: checks macOS 26.2+, chip, memory,
  GPU wired limit, `rdma_ctl`, RDMA verbs devices, a connected Thunderbolt 5 link and the
  Thunderbolt Bridge service.
- `docs/style-profile.md`: TigerStyle project profile with limits and threat model.
- CI on macOS. One tag, `v<version>`, publishes both packages to PyPI and crates.io through
  trusted publishing, with actions pinned by hash.

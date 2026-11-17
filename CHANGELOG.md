# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.0.1] - 2026-11-17

### Added

- `kvrail doctor [--json] [--strict]` in Python and Rust: checks macOS 26.2+, chip, memory,
  GPU wired limit, `rdma_ctl`, RDMA verbs devices, a connected Thunderbolt 5 link and the
  Thunderbolt Bridge service.
- `docs/style-profile.md`: TigerStyle project profile with limits and threat model.
- CI on macOS. One tag, `v<version>`, publishes both packages to PyPI and crates.io through
  trusted publishing, with actions pinned by hash.

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .doctor import exit_code, render_json, render_text, run_checks

# Exit code for a usage error, the same as argparse uses for an unknown flag.
EXIT_CODE_USAGE = 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kvrail",
        description="KVRail: RDMA KV-cache store for Apple Silicon",
    )
    parser.add_argument("--version", action="version", version=f"kvrail {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser(
        "doctor",
        help="check this Mac for RDMA-over-Thunderbolt readiness",
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.add_argument("--strict", action="store_true", help="treat warnings as failures")
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        # A missing command must not look like a passing check to a script.
        parser.print_usage(sys.stderr)
        return EXIT_CODE_USAGE
    if arguments.command != "doctor":
        raise AssertionError(f"unhandled command {arguments.command}")
    checks = run_checks()
    if arguments.json:
        print(render_json(checks))
    else:
        print(render_text(checks))
    return exit_code(checks, strict=arguments.strict)

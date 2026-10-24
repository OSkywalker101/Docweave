"""Fail CI on *new* type errors, not on the ones already known about.

The codebase has a residual set of mypy findings that are mostly pydantic union
variance: ``body: list[AnyNode]`` is the IR's declared type, and the adapters
append ``Paragraph`` and ``Table`` to it, which is correct and which mypy cannot
express without the union being covariant. Those are not going away.

Suppressing the whole module would be worse than useless -- it would also hide a
genuine mistake in a line nobody was looking at. So instead the known findings
are recorded, and this check fails only when a *new* one appears.

Line numbers are deliberately excluded from the fingerprint: moving code should
not require regenerating the baseline, but changing what it does should.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent / "tools" / "mypy-baseline.txt"
#: mypy's ``files`` setting resolves against the working directory, so this has
#: to be the repository root. Running from tools/ silently checks nothing at all
#: and reports success -- the worst possible failure mode for a type check.
ROOT = BASELINE.parent.parent
#: file | error code | message, with the line number dropped.
FINGERPRINT = re.compile(r"^(?P<file>[^:]+):\d+:\s+error:\s+(?P<rest>.*)$")


def current_findings() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--no-pretty"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    found: set[str] = set()
    for line in result.stdout.splitlines():
        match = FINGERPRINT.match(line.strip())
        if not match:
            continue
        rest = match.group("rest")
        code = rest.rsplit("[", 1)[-1].rstrip("]").strip() if "[" in rest else ""
        message = rest.rsplit("[", 1)[0].strip() if "[" in rest else rest
        found.add(f"{match.group('file')} | {code} | {message}")
    return found


def main() -> int:
    found = current_findings()

    if "--write" in sys.argv:
        header = (
            "# Known mypy findings, recorded so CI fails only on NEW type errors.\n"
            "#\n"
            "# Nearly all of these are pydantic union variance: Document.body is\n"
            "# declared list[AnyNode], and adapters append Paragraph or Table to it.\n"
            "# That is correct at runtime and inexpressible without a covariant\n"
            "# union, so it is recorded rather than silenced module by module.\n"
            "#\n"
            "# Regenerate after an intentional change with:\n"
            "#     python tools/check_types.py --write\n"
        )
        BASELINE.write_text(header + "\n".join(sorted(found)) + "\n", encoding="utf-8")
        print(f"wrote {len(found)} finding(s) to {BASELINE.name}")
        return 0

    known = set()
    if BASELINE.exists():
        known = {
            line.strip()
            for line in BASELINE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

    new = found - known
    fixed = known - found

    print(f"known findings: {len(known)}   current: {len(found)}")
    if fixed:
        print(f"\n{len(fixed)} finding(s) no longer occur -- regenerate the baseline:")
        for item in sorted(fixed):
            print(f"  - {item}")
    if new:
        print(f"\n{len(new)} NEW type error(s):")
        for item in sorted(new):
            print(f"  + {item}")
        print(
            "\nFix these, or -- if one is genuinely unavoidable -- record it in\n"
            "tools/mypy-baseline.txt with a comment saying why."
        )
        return 1
    print("\nno new type errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

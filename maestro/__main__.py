"""
Maestro entry point — ``python -m maestro <command>``.

Works from any directory when UBIK_ROOT is set or the platform default
path exists.  Typical usage::

    python -m maestro status
    python -m maestro health
    python -m maestro dashboard
    python -m maestro watch --auto-restart
    python -m maestro start
    python -m maestro shutdown --dry-run
    python -m maestro logs --follow
    python -m maestro metrics

Run ``python -m maestro --help`` for the full command list.
"""

from maestro.cli import cli

if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""
Maestro — Remote Execution Layer

Runs shell commands on the Somatic node from the Hippocampal node over SSH,
so that a single Maestro instance (running on Hippocampal) can start, stop and
otherwise manage services that physically live on Somatic.

Why this exists:
    Somatic is a separate physical machine (PowerSpec, Windows host + WSL2
    Linux guest).  vLLM and WhisperX run inside the WSL2 guest.  Reaching that
    guest non-interactively is a three-layer problem:

        macOS ssh → Windows OpenSSH (default shell) → ``wsl`` → bash

    The user's ``~/.ssh/config`` forces ``RequestTTY yes`` and
    ``RemoteCommand wsl ~`` for the ``windows-server`` host (great for
    interactive use, fatal for automation — you get "Cannot execute
    command-line and remote command").  We override both.

Command delivery:
    Rather than fight cmd.exe → wsl → bash quoting, the bash script is fed to
    the remote ``bash -s`` over **stdin**.  ssh forwards our stdin to the remote
    process, so the script travels as raw bytes and no shell on the path tries
    to re-parse it.  The only thing on the command line is the literal
    ``wsl bash -s`` (no user data), which quotes cleanly everywhere.

Public API:
    RemoteExecutor            — run(script), check() over SSH
    RemoteResult              — (returncode, stdout, stderr) container

Usage::

    from maestro.remote import RemoteExecutor
    rx = RemoteExecutor.from_config(get_config())
    if await rx.check():
        res = await rx.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader")
        print(res.stdout)

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# SSH options that make a non-interactive command possible on the
# ``windows-server`` host despite its interactive-only ~/.ssh/config:
#   BatchMode=yes       — never prompt for a password (fail fast instead)
#   RequestTTY=no       — override the config's "RequestTTY yes"
#   RemoteCommand=none  — override the config's "RemoteCommand wsl ~"
_BASE_SSH_OPTS: tuple[str, ...] = (
    "-o", "BatchMode=yes",
    "-o", "RequestTTY=no",
    "-o", "RemoteCommand=none",
)


@dataclass
class RemoteResult:
    """Outcome of a single remote command execution.

    Attributes:
        returncode: Exit code of the remote command (or of ssh itself when the
            connection failed before the command ran, e.g. 255).
        stdout: Captured standard output (decoded).
        stderr: Captured standard error (decoded).  SSH banners/warnings land
            here, never in :attr:`stdout`.
        connected: ``True`` when ssh established a session and ran the payload;
            ``False`` on connection failure/timeout.
    """

    returncode: int
    stdout: str
    stderr: str
    connected: bool


class RemoteExecutor:
    """Runs bash scripts on the Somatic node over SSH.

    Args:
        ssh_host: SSH host alias or address of the Somatic Windows host
            (e.g. ``"windows-server"``).
        wsl: When ``True``, wrap the remote payload in ``wsl bash -s`` so it
            runs inside the WSL2 Linux guest.  When ``False``, run
            ``bash -s`` directly (native Linux host).
        connect_timeout: Seconds for ssh's ``ConnectTimeout`` — how long to
            wait for the TCP/SSH handshake before giving up.
    """

    def __init__(
        self,
        ssh_host: str,
        *,
        wsl: bool = True,
        connect_timeout: float = 8.0,
    ) -> None:
        self._ssh_host = ssh_host
        self._wsl = wsl
        self._connect_timeout = connect_timeout

    @classmethod
    def from_config(cls, cfg) -> "RemoteExecutor":
        """Build a :class:`RemoteExecutor` for the Somatic node from config.

        Args:
            cfg: An :class:`~maestro.config.AppConfig` instance.

        Returns:
            Configured :class:`RemoteExecutor` targeting the Somatic node.
        """
        somatic = cfg.somatic
        return cls(
            somatic.ssh_host,
            wsl=somatic.use_wsl,
            connect_timeout=somatic.ssh_connect_timeout,
        )

    @property
    def ssh_host(self) -> str:
        """The SSH host alias this executor targets."""
        return self._ssh_host

    def _ssh_argv(self) -> list[str]:
        """Build the ssh argv that reads a bash script from stdin.

        The final positional argument is the literal remote command
        (``wsl bash -s`` or ``bash -s``) — it contains no user data, so it is
        safe from any intermediate shell's quoting rules.

        Returns:
            Argument vector suitable for ``asyncio.create_subprocess_exec``.
        """
        remote_cmd = "wsl bash -s" if self._wsl else "bash -s"
        return [
            "ssh",
            *_BASE_SSH_OPTS,
            "-o", f"ConnectTimeout={int(self._connect_timeout)}",
            self._ssh_host,
            remote_cmd,
        ]

    async def run(self, script: str, *, timeout: float = 60.0) -> RemoteResult:
        """Execute a bash *script* on the remote node.

        The script is delivered over stdin to ``bash -s`` (see module docstring)
        so multi-line scripts with arbitrary quoting work verbatim.

        Args:
            script: Bash script text.  Runs under a non-login, non-interactive
                ``bash -s``; use absolute paths or ``source`` as needed.
            timeout: Wall-clock seconds before the ssh process is killed and a
                disconnected :class:`RemoteResult` is returned.

        Returns:
            :class:`RemoteResult`.  Never raises — connection failures and
            timeouts are reported via ``connected=False`` and a non-zero
            ``returncode``.
        """
        argv = self._ssh_argv()
        logger.debug("remote: running script on %s (%d bytes)", self._ssh_host, len(script))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # ssh binary missing, etc.
            logger.warning("remote: failed to spawn ssh: %s", exc)
            return RemoteResult(returncode=255, stdout="", stderr=str(exc), connected=False)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(script.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("remote: command on %s timed out after %.0fs", self._ssh_host, timeout)
            return RemoteResult(
                returncode=124,
                stdout="",
                stderr=f"remote command timed out after {timeout:.0f}s",
                connected=False,
            )

        rc = proc.returncode or 0
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        # ssh returns 255 for its own connection-level failures.
        connected = rc != 255
        if not connected:
            logger.warning("remote: ssh connection to %s failed: %s", self._ssh_host, stderr.strip()[:200])
        return RemoteResult(returncode=rc, stdout=stdout, stderr=stderr, connected=connected)

    async def check(self, *, timeout: float = 10.0) -> bool:
        """Verify the remote node is reachable and can run bash.

        Args:
            timeout: Seconds to allow for the round-trip.

        Returns:
            ``True`` when a sentinel echo round-trips successfully.
        """
        res = await self.run("echo MAESTRO_REMOTE_OK", timeout=timeout)
        return res.connected and "MAESTRO_REMOTE_OK" in res.stdout

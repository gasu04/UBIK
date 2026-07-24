# Layer B — WSL VM boot resilience (Somatic, Windows-side)

Keeps the Somatic WSL VM (Ubuntu) itself alive and auto-starting, independent
of whatever is running inside it. Layers A and C (systemd `Restart=on-failure`
for `ubik-vllm`, health-check probes) only help *while the VM is up*. Layer B
covers the two things below it:

1. the VM comes up automatically when Windows boots, and
2. it isn't torn down mid-work by the `p9io`/`CheckConnection` bug.

Everything in this directory is version-controlled but **must be installed
manually on the Somatic Windows host** — Task Scheduler runs at the Windows
level and needs admin rights; none of this can be done from inside WSL or
from Hippocampal.

## Status as of 2026-07-23

B.1 (systemd + linger inside WSL) and most of B.4 (`.wslconfig`) were **already
in place**, confirmed directly on the box:

- `/etc/wsl.conf` already has `[boot] systemd=true` and `[user] default=gasu`.
- `loginctl show-user gasu` already reports `Linger=yes`.
- `/mnt/c/Users/gasu.Adrian/.wslconfig` already has `vmIdleTimeout=-1` and
  `networkingMode=NAT` (do not change `networkingMode` — mirrored mode was
  previously linked to the `CheckConnection` poweroff and a port-8002
  collision).

So no `.wslconfig` or `wsl.conf` edits are required by this task. B.2 and B.3
(the Scheduled Tasks below) are net-new and are what's left to install.

**Known gap, unrelated to Layer B:** `maestro/` is currently missing from the
repo checkout and there is no `ubik-vllm` systemd unit or running vLLM
process — only `whisperx.service` exists. This means the Layer B verification
steps that assume `ubik-vllm` exists (steps 1 and 3 below) will not pass until
that's addressed separately; the VM-boot mechanics themselves can still be
installed and verified independently (step 2, and step 4's log).

## Files

| File | Purpose |
|---|---|
| `ubik-wsl-boot.ps1` | Boots the WSL VM at Windows startup (B.2) |
| `ubik-wsl-keepalive.ps1` | Heartbeats the VM every 2 min, re-boots on failure, logs poweroffs (B.3) |
| `ubik-wsl-boot-task.xml` | Task Scheduler definition for the boot task |
| `ubik-wsl-keepalive-task.xml` | Task Scheduler definition for the keepalive task |

## Install

### 1. Copy the scripts to a native Windows path

Task Scheduler's boot trigger fires **before the WSL VM exists**, so it
cannot read the scripts via `\\wsl.localhost\Ubuntu\...` (that path only
resolves once the VM is already up — chicken-and-egg). Copy them to a stable
Windows-native path first, and re-copy after every `git pull` that touches
this directory:

```powershell
New-Item -ItemType Directory -Force -Path C:\UBIK\somatic\windows | Out-Null
Copy-Item \\wsl.localhost\Ubuntu\home\gasu\ubik\somatic\windows\*.ps1 C:\UBIK\somatic\windows\
```

(Run this once the VM is up — e.g. right after a normal login — not as part
of the boot task itself.)

### 2. Register the two Scheduled Tasks

Simplest path — `schtasks.exe` from an elevated PowerShell/cmd prompt. This
prompts interactively for the `gasu` Windows account password; it is not
stored in any repo file:

```powershell
schtasks /Create /TN "UBIK\ubik-wsl-boot" `
  /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\UBIK\somatic\windows\ubik-wsl-boot.ps1" `
  /SC ONSTART /DELAY 0000:30 /RU gasu /RL HIGHEST /IT

schtasks /Create /TN "UBIK\ubik-wsl-keepalive" `
  /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\UBIK\somatic\windows\ubik-wsl-keepalive.ps1" `
  /SC MINUTE /MO 2 /RU gasu /RL HIGHEST /IT
```

`/IT` ("Interactive Token") lets the task run without a stored password as
long as `gasu` has logged into this machine before, but it **won't run when
no user is logged in** — which defeats the point of a headless boot task. For
a true "run whether logged on or not" task, either:

- Answer the password prompt when `schtasks` asks (omit `/IT`, add
  `/RP <password>` or let it prompt), or
- Import the XML definitions instead, which is the more reliable path for
  unattended boot tasks:

```powershell
schtasks /Create /TN "UBIK\ubik-wsl-boot" /XML C:\UBIK\somatic\windows\ubik-wsl-boot-task.xml
schtasks /Create /TN "UBIK\ubik-wsl-keepalive" /XML C:\UBIK\somatic\windows\ubik-wsl-keepalive-task.xml
```

The XML files use `LogonType=S4U` (`RunLevel=HighestAvailable`), which runs
the task as `gasu` without storing a password, provided `gasu` has logged
into the machine at least once and has the "Log on as a batch job" right
(Task Scheduler grants this automatically on import/creation). If Task
Scheduler complains about the account when importing, open the task in
Task Scheduler GUI → General tab → re-select the `gasu` user via "Change
User or Group" and save.

### 3. Optional B.4 hardening

If WSLg/GUI apps are unused, add to `.wslconfig`:

```ini
[wsl2]
guiApplications=false
```

Then run `wsl --shutdown` once from the Windows host (not from inside WSL —
this terminates the running VM, including any current SSH/agent session) and
let the B.2 boot task bring it back up cleanly.

## Verification (acceptance test for Layer B)

Run these from the Windows host, not from inside WSL:

1. `wsl --shutdown` → within the keepalive interval (≤2 min) the VM is back
   and `curl http://<tailscale-somatic-ip>:8002/health` returns 200, with no
   human action. *(Currently blocked on the missing `ubik-vllm` unit — see
   "Known gap" above; the VM itself will still come back up.)*
2. Full Windows reboot → after boot, `whisperx` (and, once restored,
   `ubik-vllm`) are healthy from Hippocampal with no login required.
3. `systemctl --user kill -s SIGKILL ubik-vllm` (once that unit exists again)
   → `Restart=on-failure` brings it back in ~10s.
4. Check `%USERPROFILE%\ubik-wsl-keepalive.log` periodically — it should
   accrue real poweroff timestamps if the `p9io` bug fires, or stay empty
   (both outcomes are useful telemetry).

## B.5 — root-cause tracking (not fixed here)

The `p9io`/`CheckConnection` VM teardown is a known WSL2 bug class, not
something fixable from this repo. When it recurs, capture:

- `%USERPROFILE%\ubik-wsl-keepalive.log` timestamp of the `VM_DOWN` entry
- Windows Event Viewer around that timestamp (Application and System logs)
- `wsl --version` output, to compare against WSL release notes for known
  fixes

Layers B.1–B.4 make the system recover regardless of whether this is ever
root-fixed upstream.

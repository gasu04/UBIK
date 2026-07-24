# UBIK Hardening Plan — Stable & Reliable vLLM / WSL / Somatic

**Author:** Claude Code session 2026-07-23 (Hippocampal)
**Status:** PROPOSAL — read fully before executing. Nothing here has been applied.
**Scope:** Make the vLLM-on-Somatic launch path robust for the long term, plus the
supporting CLAUDE.md-compliance and venv-isolation work uncovered in the same audit.
**Primary target (per request):** the WSL → vLLM link and the Somatic node.

---

## 0. Read this first — the one-paragraph thesis

The vLLM start failure is **endemic because the launch architecture is fragile at three
layers and maestro is blind to two of them.** The proximate cause has differed every single
recurrence (idle-timeout teardown, `p9io` VM poweroff, mirrored-mode port collision,
orphaned relay socket, direct-`os.environ` bug, clean VM teardown mid-load). That variability
is the tell: we have not been fixing *a* bug, we have been repeatedly re-diagnosing a system
that **cannot keep its own runtime alive and cannot tell us when it dies.** The durable fix is
therefore not another patch to one cause — it is to (A) make the vLLM unit **persistent and
self-healing** instead of transient, (B) keep the **WSL VM itself alive and auto-starting** on
the Somatic Windows host, and (C) give maestro **eyes** so any future death is reported
instantly with the journal tail instead of a silent 300 s timeout. D and E remove the
remaining CLAUDE.md violations and the cross-repo venv interference.

**Layer independence matters:** A, C, D, E are all Hippocampal-side / repo code and can be
built and tested here. **B is the only part that requires admin on the Somatic Windows host**
and cannot be done from this node — it is called out explicitly wherever it appears.

---

## 1. Current architecture (verified this session, not from memory)

```
Hippocampal (Mac Mini M4, macOS)                 Somatic (PowerSpec, Windows host)
┌───────────────────────────────┐                ┌────────────────────────────────────────┐
│ maestro  (runs in DeepSeek     │   ssh          │ Windows OpenSSH server                  │
│  venv — NOT its own venv)      │ ─────────────▶ │   └─ wsl bash -s ──▶ WSL2 Ubuntu guest  │
│                                │  windows-server │        └─ systemd --user (needs linger) │
│ RemoteExecutor.run(script)     │  100.75.228.46  │             └─ systemd-run --user       │
│ VllmService._remote_start()    │                │                  --unit=ubik-vllm        │
│ base._wait_for_healthy()       │  probe /health │                  (TRANSIENT unit)        │
│   → GET Tailscale:8002/health  │ ◀───────────── │        vllm_server.py wrapper            │
└───────────────────────────────┘   Tailscale     │        venv: pytorch_env_vllm024 (0.24) │
                                                   └────────────────────────────────────────┘
```

**Established facts (file:line):**
- `maestro/services/vllm_service.py:519` `_remote_start` — builds a bash script that runs
  `systemd-run --user --unit=ubik-vllm ... "$PYTHON" "$SERVER" ...`. The unit is **transient**
  (in-memory only; `man systemd-run`). It is **not `enable`d** and has **no `Restart=`**.
- `maestro/services/vllm_service.py:570-578` — `--property=KillSignal=SIGTERM`,
  `KillMode=mixed`, `TimeoutStopSec=90`, `--setenv` for the Blackwell vars. Correct, but
  re-declared on every start instead of living in an installed unit file.
- `maestro/remote.py:139` — remote command is literally `wsl bash -s`; script travels over
  stdin (good — avoids cmd.exe/quoting hell). Overrides the interactive `RequestTTY yes` /
  `RemoteCommand wsl ~` from `~/.ssh/config`.
- `maestro/services/base.py:290-318` `_wait_for_healthy` — polls `probe()` at a **constant 3 s**
  interval for the full `max_wait_s` (300 s for vLLM), then logs a generic
  `"<name> did not become healthy within <N>s"`. **Never** runs `systemctl --user is-active`
  or reads the journal. This is the diagnostic blind spot.
- `maestro/config.py:262` `vllm_venv="/home/gasu/pytorch_env_vllm024"` (0.24 active),
  `pytorch_env` (0.13) retained as rollback. Somatic venv isolation is **correct**.
- `~/.ssh/config` — `windows-server` (RemoteCommand `wsl ~`, for automation via the override)
  and `windows-plain` (RemoteCommand none → Windows shell, for admin). Both present.
- Hippocampal: `venv → /home/gasu/pytorch_env` is a **dead symlink** (Linux path on macOS);
  `maestro` alias hardcodes the **DeepSeek** interpreter. No dedicated UBIK venv exists.

**Why the transient unit is the structural weakness:** `systemd-run --user` registers a unit
that exists only in the running user-manager's memory. The moment the WSL VM is torn down
(`wsl --shutdown`, a Windows update, Docker Desktop's WSL integration cycling, or the `p9io`
poweroff bug), the unit is gone — not stopped-and-restartable, *gone*. On the next VM boot
nothing brings vLLM back, and maestro only finds out 300 s later with no reason attached.

---

## 2. The fix, in five layers

Ordered by **value ÷ risk**. Layer C first (pure diagnostic, zero behavior change, highest
value). Then A (persistence). Then B (the Somatic/WSL keepalive — the deepest fix, but needs
the Windows host). Then D, E (compliance + isolation).

---

### Layer C — Give maestro eyes (do this FIRST: highest value, lowest risk)

**Goal:** any future vLLM death is reported **instantly with the journal tail**, instead of a
silent 300 s timeout. This alone would have collapsed every past multi-hour forensic dig into
a one-line error. It changes no launch behavior — it only observes.

**Design (surgical, service-agnostic base + vLLM-specific override):**

1. Add an optional liveness hook to the base class — default is a no-op, so no other service
   changes behavior:
   ```python
   # maestro/services/base.py  (UbikService)
   async def _liveness_diagnostic(self) -> Optional[str]:
       """Return None if the underlying process/unit is alive; else a short
       diagnostic string (e.g. journal tail) explaining why it died.

       Default: None (no out-of-band liveness signal available). Services whose
       process can die independently of the health port override this.
       """
       return None
   ```

2. In `_wait_for_healthy`, after each unhealthy probe, consult the hook and **abort early**
   with the diagnostic if the process is confirmed dead:
   ```python
   if not result.healthy:
       diag = await self._liveness_diagnostic()
       if diag is not None:
           logger.error("%s died during startup wait (%ds): %s",
                        self.name, elapsed, diag)
           return False
   ```
   (Poll the hook at most every ~9 s, not every 3 s, to avoid an extra SSH round-trip each cycle.)

3. Implement the hook in `VllmService` for the **remote** path only — one cheap SSH call, and
   the journal tail only when the unit is actually dead:
   ```python
   async def _liveness_diagnostic(self) -> Optional[str]:
       if not self._is_remote():
           return None
       script = f'''{_USER_SYSTEMD_ENV}
   state=$(systemctl --user is-active {_VLLM_UNIT} 2>/dev/null || echo unknown)
   echo "STATE=$state"
   if [ "$state" != "active" ] && [ "$state" != "activating" ]; then
       journalctl --user -u {_VLLM_UNIT} -n 30 --no-pager 2>/dev/null | tail -n 30
   fi'''
       res = await self._remote.run(script, timeout=15.0)
       if not res.connected:
           return None                      # can't tell — keep waiting, don't false-alarm
       if "STATE=active" in res.stdout or "STATE=activating" in res.stdout:
           return None                      # alive, still loading — keep waiting
       return res.stdout.strip()[:2000]     # dead — return journal tail
   ```
   **Privacy (§2.4):** the journal tail is vLLM engine output (model load, CUDA, backend
   selection) — not memory content. Safe to log. Do **not** widen this to dump user prompts.

**Files touched:** `maestro/services/base.py` (add hook + one call site),
`maestro/services/vllm_service.py` (override). ~40 lines total.

**Test (§2.6 — this is a Tier-1-adjacent silent-failure fix):**
- `test_wait_for_healthy_aborts_on_unit_death` — mock `_liveness_diagnostic` to return a
  journal string on the 2nd poll; assert `_wait_for_healthy` returns `False` in <30 s (not
  300 s) and the diagnostic is logged.
- `test_wait_for_healthy_keeps_waiting_while_activating` — hook returns `None`; assert it
  polls to the healthy result.
- `test_liveness_none_when_ssh_down` — `connected=False` must return `None` (never false-alarm
  on a transient SSH blip).

**Rollback:** revert two files. No state to undo.

---

### Layer A — Persistent, self-healing vLLM unit (replace transient `systemd-run`)

**Goal:** the vLLM unit survives as an **installed, enabled** systemd user service with
`Restart=on-failure`, so that *within a running VM* a crash or hiccup auto-recovers, and on
VM boot the unit starts itself (given linger from Layer B).

**Design:**

1. **Add the unit file to the repo** as the single source of truth (version-controlled,
   reviewable, no more ad-hoc `--property` flags):
   ```ini
   # somatic/systemd/ubik-vllm.service   (installed to ~/.config/systemd/user/ on Somatic)
   [Unit]
   Description=UBIK vLLM inference server (Somatic)
   After=default.target

   [Service]
   Type=simple
   # %h expands to the user home on the Somatic node — no hardcoded /home/gasu.
   Environment=VLLM_FLASH_ATTN_VERSION=2
   Environment=PYTORCH_ALLOC_CONF=expandable_segments:True
   Environment=CUDA_MODULE_LOADING=LAZY
   Environment=VLLM_USE_FLASHINFER_SAMPLER=0
   ExecStart=%h/pytorch_env_vllm024/bin/python %h/ubik/somatic/inference/vllm_server.py \
       --rtx5080 --skip-checks --config %h/ubik/config/models/vllm_config.yaml \
       --model %h/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ --port 8002
   KillSignal=SIGTERM
   KillMode=mixed
   TimeoutStopSec=90
   Restart=on-failure
   RestartSec=10
   StartLimitIntervalSec=600
   StartLimitBurst=4

   [Install]
   WantedBy=default.target
   ```
   **Note the env/paths must stay in sync with `maestro/config.py`.** To avoid drift, maestro
   should *render* this file from config (below), OR the file is treated as canonical and
   config reads from it. Recommended: **maestro renders it** so `VLLM_MODEL_PATH` / `VLLM_PORT`
   / `VLLM_VENV_PATH` remain the single source (CLAUDE.md §2.1 — config drives everything).

2. **Change `_remote_start` from `systemd-run` to install-then-start** (idempotent sync):
   - Write the rendered unit to `~/.config/systemd/user/ubik-vllm.service` over SSH (only if
     content changed — compare a hash first to avoid needless `daemon-reload`).
   - `systemctl --user daemon-reload` (only on change).
   - `systemctl --user enable ubik-vllm` (idempotent).
   - `systemctl --user restart ubik-vllm`.
   - Keep the existing pre-flight guards (`ALREADY_ACTIVE`, `ALREADY_RUNNING_UNMANAGED`,
     `reset-failed`).
   - `_remote_stop` changes from `stop` on a transient unit to `stop` on the installed unit —
     already compatible (it calls `systemctl --user stop ubik-vllm`).

3. **Retire the transient path.** Once installed-unit start is proven, delete the
   `systemd-run` branch so there is exactly one launch mechanism.

**Why `Restart=on-failure` is safe here:** the `StartLimitBurst=4 / IntervalSec=600` cap means
a genuinely broken start (bad model path, VRAM exhausted) fails 4× then stops — it will **not**
thrash forever. A transient VM hiccup mid-load, however, now recovers automatically. Combined
with Layer C, if it *does* hit the start-limit, maestro reports the journal tail instantly.

**Files touched:** new `somatic/systemd/ubik-vllm.service` (or a renderer in
`vllm_service.py`), `maestro/services/vllm_service.py` `_remote_start`/`_remote_stop`.

**Test (§2.6):**
- Unit-file renderer: `test_render_vllm_unit_uses_config_values` — assert model path, port,
  venv, and all four Blackwell env lines come from config, no hardcoded `/home/gasu`.
- `_remote_start` mocked-SSH tests: install-on-change, skip-reload-when-unchanged,
  enable+restart issued, `ALREADY_ACTIVE` short-circuit.

**Rollback:** `systemctl --user disable --now ubik-vllm; rm ~/.config/systemd/user/ubik-vllm.service`
on Somatic returns to the transient behavior; revert the maestro change.

---

### Layer B — Keep the WSL VM alive & auto-starting (THE Somatic fix — needs Windows admin)

**This is the deepest layer and the one the request calls out.** Layers A and C make vLLM
recover *while the VM is up*. But **if the WSL VM powers off, there is no systemd at all** —
no enabled unit, no linger, nothing. So the durable Somatic fix has two independent jobs:

> **(1) make the VM come up automatically, and (2) keep it from being torn down mid-work.**

**⚠️ Every step in Layer B runs on the Somatic *Windows* host and requires admin. It cannot be
done from Hippocampal. Use the `windows-plain` SSH alias (lands in the Windows shell) or do it
at the machine. These are one-time setup steps, version-controlled under `somatic/windows/`.**

**B.1 — Enable systemd + user lingering inside WSL (the linchpin).**
- `/etc/wsl.conf` (inside the WSL guest) must contain:
  ```ini
  [boot]
  systemd=true
  [user]
  default=gasu
  ```
- `loginctl enable-linger gasu` (inside WSL, once). **This is what makes the enabled
  `ubik-vllm` unit start at VM boot without an SSH login.** Verify with
  `loginctl show-user gasu | grep Linger` → `Linger=yes`.
- Without linger, the user systemd manager (and thus vLLM) only exists while an SSH session is
  open — exactly the fragility we're removing.

**B.2 — Auto-start the VM when Windows boots (Scheduled Task).**
- The VM only boots when *something* invokes `wsl`. Create a Windows Scheduled Task
  (`somatic/windows/ubik-wsl-boot.ps1` + a Task Scheduler XML) that runs **at system startup**
  (not logon — survives headless reboots), as the `gasu` user, highest privileges:
  ```powershell
  # somatic/windows/ubik-wsl-boot.ps1
  wsl.exe -d Ubuntu -u gasu -e /bin/true    # boots the VM; linger+enable brings up vLLM
  ```
- Result: Windows boots → task fires → VM boots → systemd user manager starts (linger) →
  `ubik-vllm` starts (enabled). Fully hands-off recovery from a machine reboot.

**B.3 — Keepalive heartbeat against the `p9io` mid-load poweroff (the intermittent killer).**
- The `p9io`/`CheckConnection` bug tears the VM down ~15 s into heavy work (model load). A
  second, independent reference on the VM makes it far less likely to be reaped, and turns a
  silent poweroff into a logged event:
  ```powershell
  # somatic/windows/ubik-wsl-keepalive.ps1  (Scheduled Task, every 2 min)
  $r = wsl.exe -d Ubuntu -u gasu -e /bin/echo alive 2>&1
  if ($LASTEXITCODE -ne 0) {
      Add-Content "$env:USERPROFILE\ubik-wsl-keepalive.log" "$(Get-Date -Format o) VM_DOWN rc=$LASTEXITCODE $r"
      wsl.exe -d Ubuntu -u gasu -e /bin/true   # attempt immediate re-boot
  }
  ```
- This gives us the **first real telemetry** on how often the VM actually dies, and
  auto-reboots it, at which point Layer A brings vLLM back.

**B.4 — Reduce the teardown surface (`.wslconfig`, Windows `%UserProfile%`).**
- Confirm/keep (already set per session history): `vmIdleTimeout=-1`, `networkingMode=NAT`.
  **Keep NAT** — mirrored mode has been associated with the `CheckConnection` poweroff and a
  port-8002 collision in past incidents. Do **not** switch to mirrored.
- Add `[wsl2] guiApplications=false` if WSLg is unused (smaller surface, fewer host sockets).
- After any `.wslconfig` change: `wsl --shutdown` once, then let B.2 re-boot cleanly.

**B.5 — Investigate the root `p9io` bug (out of repo scope, tracked).**
- Capture `dmesg`/Windows Event Log around a poweroff (B.3's log gives timestamps). Check WSL
  version (`wsl --version`) against known-fixed builds; a WSL upgrade may retire the bug
  outright. Track as a standing item; B.1–B.4 make the system robust *regardless* of whether
  this is ever root-fixed.

**Files (repo, tracked; installed manually on Windows):**
`somatic/windows/ubik-wsl-boot.ps1`, `somatic/windows/ubik-wsl-keepalive.ps1`,
`somatic/windows/*.xml` (Task Scheduler definitions), `somatic/windows/README.md`
(exact `schtasks`/Task Scheduler install commands + verification).

**Verification of Layer B (the acceptance test for "robust"):**
1. `wsl --shutdown` on Somatic → within ≤ (task interval) the VM is back and
   `curl Tailscale:8002/health` → 200 **with no human action**.
2. Full Windows reboot → after boot, vLLM is healthy from Hippocampal with no login.
3. `systemctl --user kill -s SIGKILL ubik-vllm` → `Restart=on-failure` brings it back in ~10 s.
4. B.3 log accrues real poweroff timestamps (or stays empty — either way we now *know*).

---

### Layer D — §2.3 resilience compliance (import canonical, never reimplement)

**Goal:** close the §2.3 FAIL. The Hippocampal→Somatic SSH and probe paths currently have no
circuit breaker and no retry+jitter — the constant-3 s poll is the exact probe-storm pattern
§2.3 forbids.

- **Import** `~/ubik/somatic/mcp_client/circuit_breaker.py` (Probe-Latch breaker) and
  `resilience.py` (retry + backoff + jitter). **Do not write new ones** (CLAUDE.md §3.4).
- Wrap `RemoteExecutor.run` (SSH) and the vLLM `probe` in the breaker: when Somatic is flapping,
  reject fast instead of hammering; in HALF_OPEN, exactly one probe.
- Replace the fixed 3 s poll cadence with backoff+jitter so recovering nodes aren't hit in
  lockstep.
- **Dependency note:** the canonical modules live in `~/ubik/somatic/mcp_client/`. maestro must
  be able to import them — this is a concrete argument for Layer E (a shared UBIK venv that has
  both maestro and the mcp_client package importable), or packaging `mcp_client` for reuse.

**Test:** breaker opens after N failed SSH attempts; HALF_OPEN admits exactly one probe;
jittered backoff sequence is bounded by `min(base*2^n, max)`.

---

### Layer E — Dedicated UBIK venv on Hippocampal (removes cross-repo interference)

**Goal:** stop maestro/ingestion/hippocampal from running inside the **DeepSeek** venv, where
`fastmcp` is force-upgraded 2.14.3 → 3.4.2 (a major version the hippocampal pin can never hold
while shared), plus `sentence-transformers` / `python-dotenv` drift.

1. Create `ubik-venv` (py3.13) = union of `maestro` + `ingestion` + `hippocampal` requirements,
   **excluding torch/langchain/whisper** (those belong to Somatic and DeepSeek respectively).
2. Resolve the `fastmcp` 2 vs 3 conflict deliberately (pin the version hippocampal actually
   needs and test the MCP client against it).
3. Replace the dead `venv → /home/gasu/pytorch_env` symlink with the real `ubik-venv` (or a
   `.python-version`), and repoint the `maestro` zsh alias:
   ```
   alias maestro='cd "/Volumes/990PRO 4T/UBIK" && "/Volumes/990PRO 4T/UBIK/ubik-venv/bin/python" -m maestro'
   ```
4. Emit per-subproject `requirements.lock` (`pip freeze`) — closes §2.7.
5. Correct `ENVIRONMENT.md` to document: DeepSeek venv (LLM app), UBIK venv (infra/maestro),
   Somatic `pytorch_env_vllm024` (vLLM), `ubik-whisperx-venv` (WhisperX) — four venvs, four
   jobs, no overlap.

**Risk:** medium — changing the interpreter maestro runs under. Mitigate by building `ubik-venv`
alongside, testing `python -m maestro status` against it, and only flipping the alias once green.
The DeepSeek venv stays untouched as instant rollback.

---

## 3. Recommended execution order

| Step | Layer | Where | Risk | Reversible? | Needs Windows admin? |
|:--|:--|:--|:--:|:--:|:--:|
| 1 | **C** diagnostic eyes | Hippocampal code | very low | trivial | no |
| 2 | **A** persistent unit | Hippocampal code + Somatic WSL | low | yes | no (WSL via SSH) |
| 3 | **B.1** systemd+linger | Somatic WSL | low | yes | no (WSL via SSH) |
| 4 | **B.2** boot task | Somatic Windows | low | yes | **yes** |
| 5 | **B.3** keepalive+telemetry | Somatic Windows | low | yes | **yes** |
| 6 | **B.4** `.wslconfig` review | Somatic Windows | low | yes | **yes** |
| 7 | **E** UBIK venv | Hippocampal | medium | yes | no |
| 8 | **D** §2.3 resilience | Hippocampal code | medium | yes | no |
| 9 | **B.5** p9io root cause | Somatic Windows | — | n/a | **yes** (investigate) |

Rationale: **C before A** so that the moment we start touching the launch path we can already
*see* failures. **A before B** so the unit is enable-able before we wire linger/boot to it.
**B.1 before B.2** (linger must exist before boot-start is meaningful). **E before D** because
D's canonical `mcp_client` import wants a venv that can see both packages.

Steps 1–3 give the biggest reliability jump (persistent + self-healing + observable *within* a
running VM). Steps 4–6 deliver true hands-off recovery across VM/host reboots — the part that
finally ends the endemic manual restarts.

---

## 4. Definition of done ("robust", measurable)

The system is considered hardened when **all** of these pass without human intervention:

1. `wsl --shutdown` on Somatic → vLLM `/health` returns 200 from Hippocampal within the
   keepalive interval, no manual step.
2. Full Somatic Windows reboot → vLLM healthy from Hippocampal after boot, no login.
3. `systemctl --user kill -s SIGKILL ubik-vllm` → auto-restart in ~10 s (`Restart=on-failure`).
4. A genuinely broken start (e.g. bad model path) fails fast and maestro prints the **journal
   tail** — no 300 s silent timeout (Layer C).
5. `maestro start --service vllm` run 5× on 5 different days succeeds first-try each time
   (the endemic failure is gone).
6. `python -m maestro status` runs under `ubik-venv`; `pip list` shows fastmcp at the
   hippocampal-pinned version, not 3.4.2.
7. CLAUDE.md self-tests: no circuit-breaker/retry FAIL (§2.3), `_wait_for_healthy` surfaces
   unit death; `grep 100.x` still clean.
8. B.3 keepalive log exists and is being written (we now have VM-poweroff telemetry).

---

## 5. Explicitly out of scope / deferred (so it isn't silently dropped)

- **§2.8** `UbikError` hierarchy + narrowing the 94 bare `except Exception` — real, but noisy
  and low-risk; schedule after the reliability work lands.
- **chromadb container `unhealthy` flag** — cosmetic (v1 healthcheck vs v2 service); fix the
  compose healthcheck to probe `/api/v2/heartbeat`. Not part of the vLLM/WSL hardening.
- **chromadb version alignment** (client 1.4.1 / native 1.3.7 / Somatic 1.5.1) — carried backlog.
- **§2.6 test-dir restructure** into unit/integration/e2e/fixtures — do opportunistically as
  each layer above adds its tests.

---

## 6. What I did NOT change

Per §1.3 (Surgical Changes): this document is a proposal only. The only code change made in the
originating session was hardening `.gitignore` to stop `chromadb_data/` (37 MB personal memory)
from being committable — unrelated to this plan but a live privacy trap that could not wait.
No maestro / ingestion / hippocampal source has been touched. Awaiting review before any of
Layers A–E is executed.
```

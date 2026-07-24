<#
.SYNOPSIS
    Heartbeats the UBIK Somatic WSL VM and re-boots it if it has gone down.
.DESCRIPTION
    Layer B.3 of the Somatic resilience plan. Runs every 2 minutes via Task
    Scheduler. Works around an intermittent p9io/CheckConnection bug (tracked
    under Layer B.5) that can tear the VM down ~15s into heavy load, such as a
    vLLM model load. A failed heartbeat is logged with a timestamp and exit
    code, then an immediate re-boot is attempted so Layer A (systemd
    Restart=on-failure) can bring vLLM back once the VM is up again.
    Installed manually on the Somatic Windows host — see README.md in this
    directory for the exact Task Scheduler setup and verification steps.
#>

$logPath = Join-Path $env:USERPROFILE "ubik-wsl-keepalive.log"

$result = wsl.exe -d Ubuntu -u gasu -e /bin/echo alive 2>&1

if ($LASTEXITCODE -ne 0) {
    $timestamp = Get-Date -Format o
    Add-Content -Path $logPath -Value "$timestamp VM_DOWN rc=$LASTEXITCODE $result"
    wsl.exe -d Ubuntu -u gasu -e /bin/true
}

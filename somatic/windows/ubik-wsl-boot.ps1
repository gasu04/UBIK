<#
.SYNOPSIS
    Boots the UBIK Somatic WSL VM (Ubuntu) at Windows system startup.
.DESCRIPTION
    Layer B.2 of the Somatic resilience plan. Runs as a Task Scheduler "At
    startup" task, before any interactive login. Invoking wsl.exe boots the
    VM if it isn't already running; systemd with linger enabled for gasu
    (Layer B.1) then starts enabled units (ubik-vllm, whisperx) with no SSH
    session required.
    Installed manually on the Somatic Windows host — see README.md in this
    directory for the exact Task Scheduler setup and verification steps.
#>

wsl.exe -d Ubuntu -u gasu -e /bin/true
exit $LASTEXITCODE

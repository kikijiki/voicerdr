# voicerdr task runner — run inside `nix develop` / direnv (provides uv, just, ruff, PortAudio).

set shell := ["bash", "-euo", "pipefail", "-c"]

repo := justfile_directory()
run := repo / "scripts/run.sh"
lifecycle := repo / "scripts/plugin-lifecycle.sh"

export UV_LINK_MODE := env_var_or_default("UV_LINK_MODE", "copy")

# List recipes
default:
    @just --list

# ---- env --------------------------------------------------------------------

# Install / refresh the Python package into .venv via uv
sync:
    cd {{quote(repo)}} && uv sync

# One-time checkout setup: deps, model caches, Herdr plugin link
setup: bootstrap-models link

# Show resolved plugin/config/state/control paths
paths: sync
    {{quote(run)}} paths

# Sanity-check tools + Herdr socket
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "python: $(python3 --version)"
    echo "uv:     $(uv --version)"
    echo "just:   $(just --version)"
    echo "ruff:   $(ruff --version)"
    command -v herdr >/dev/null && herdr --version || echo "herdr:  missing from PATH"
    sock="${HERDR_SOCKET_PATH:-$HOME/.config/herdr/herdr.sock}"
    if [[ -S "$sock" ]]; then
      echo "herdr socket: $sock (ok)"
    else
      echo "herdr socket: $sock (missing — is Herdr running?)"
    fi
    cd {{quote(repo)}} && uv run python -c "import voicerdr; print('voicerdr', voicerdr.__version__)"

# ---- daemon -----------------------------------------------------------------

# Sync, ensure the daemon, then unmute / start listening
start: sync
    {{quote(run)}} ensure
    {{quote(run)}} ctl listen

# Gracefully shut down the daemon (and stop listening)
stop: sync
    {{quote(run)}} ctl quit

# Idempotently start the out-of-band daemon for this Herdr session
ensure: sync
    {{quote(run)}} ensure

# Daemon health / mute mode / subscribe flags
status: sync
    {{quote(run)}} ctl status

# Unmute mic and start Moonshine STT (waits until pipeline is ready)
listen: sync
    {{quote(run)}} ctl listen

# Mute mic / stop STT
mute: sync
    {{quote(run)}} ctl mute

# Graceful daemon shutdown
quit: sync
    {{quote(run)}} ctl quit

# Provision Moonshine/Kokoro/NLTK caches now for later offline operation
bootstrap-models: sync
    {{quote(run)}} bootstrap-models

# Install, enable, and start optional systemd --user supervision
service-install: sync
    {{quote(run)}} service install --enable --now

# Show optional systemd unit and daemon health
service-status: sync
    {{quote(run)}} service status

# Disable supervision and guarantee the daemon has quit
service-disable: sync
    {{quote(run)}} service disable

# Quit and remove the voicerdr-managed systemd user unit
service-uninstall: sync
    {{quote(run)}} service uninstall

# Resolve a workspace label/alias to an agent target (no prompt)
resolve space: sync
    {{quote(run)}} ctl resolve --space {{quote(space)}}

# Simulate a spoken transcript (wake + command) without the mic
# Example: just ingest "hey jenny tell voice reply ACK then stop"
ingest text: sync
    {{quote(run)}} ctl ingest --text {{quote(text)}}

# Speak a line via Kokoro TTS (daemon must be running)
say text: sync
    {{quote(run)}} ctl say --text {{quote(text)}}

# Show space # / nicknames (also Herdr action voicerdr: spaces)
aliases: sync
    {{quote(run)}} ctl aliases

# List PyAudio input devices (for audio.input_device_index)
devices: sync
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{quote(repo)}}
    uv run python - <<'PY'
    import pyaudio
    p = pyaudio.PyAudio()
    print(f"default_input={p.get_default_input_device_info().get('index')}")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        ch = int(info.get("maxInputChannels") or 0)
        if ch < 1:
            continue
        print(f"{i}: {info.get('name')} (in={ch}, rate={info.get('defaultSampleRate')})")
    p.terminate()
    PY

# ---- herdr plugin -----------------------------------------------------------

# Link this checkout into Herdr
link:
    herdr plugin link {{quote(repo)}}

# Safely quit, then unlink the voicerdr plugin
unlink:
    {{quote(lifecycle)}} unlink

# Safely enable + ensure / quit + disable the Herdr plugin
plugin-enable:
    {{quote(lifecycle)}} enable

plugin-disable:
    {{quote(lifecycle)}} disable

# Invoke ensure via Herdr (uses Herdr-injected STATE/CONFIG dirs)
herdr-ensure:
    herdr plugin action invoke voicerdr.ensure

# Invoke status via Herdr
herdr-status:
    herdr plugin action invoke voicerdr.status

# ---- quality ----------------------------------------------------------------

# Fast unit and regression checks
test: sync
    cd {{quote(repo)}} && uv run python -m unittest discover -s tests -v

# Lint Python with ruff
lint: sync
    cd {{quote(repo)}} && ruff check src tests

# Format Python with ruff
fmt: sync
    cd {{quote(repo)}} && ruff format src tests && ruff check --fix src tests

# Format flake.nix
fmt-nix:
    nixfmt flake.nix

# ---- misc -------------------------------------------------------------------

# Remove local venv (flake tools stay; next `just sync` recreates)
clean:
    rm -rf -- {{quote(repo / ".venv")}}

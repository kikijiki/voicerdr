# voicerdr

Local voice assistant for [Herdr](https://herdr.dev). Default name is Jenny.
Moonshine speech-to-text, a user-supplied OpenAI-compatible LLM, and Kokoro TTS
route spoken messages into Herdr spaces and announce useful agent transitions.

## Requirements

Linux, Herdr 0.8+, Python 3.11+, and PortAudio. You must also run a local
OpenAI-compatible chat-completions server; voicerdr installs neither the LLM
server nor any model files.

## Installation

### With Nix (recommended)

```bash
git clone https://github.com/kikijiki/voicerdr.git voicerdr
cd voicerdr
direnv allow                 # or: nix develop
just sync
just bootstrap-models        # once online; prepares later offline use
just link
herdr plugin action invoke voicerdr.ensure
herdr plugin action invoke voicerdr.listen
```

`scripts/run.sh` enters the locked flake runtime once per session and reuses it
until the flake changes.

### Without Nix

Install `uv`, Python, `pkg-config`, PortAudio development/runtime libraries, and
libsndfile before `uv sync`:

| Distribution | Native packages |
| --- | --- |
| Debian/Ubuntu | `portaudio19-dev libsndfile1 pkg-config python3-dev` |
| Fedora | `portaudio-devel libsndfile-devel pkgconf-pkg-config python3-devel` |
| Arch Linux | `portaudio libsndfile pkgconf` |

Exact package names vary by release. PortAudio must be visible when PyAudio is
installed and when the daemon opens input/output devices.

## Configuration

On first ensure, example config is copied (without overwriting) to
`~/.config/herdr/plugins/config/voicerdr/`. Edit the seeded config and set your
LLM endpoint:

```toml
[llm]
base_url = "http://127.0.0.1:8080/v1"
api_key = "local"
model = "the-model-name-served-by-your-endpoint"
```

Point `[llm].base_url` and `[llm].model` at your local OpenAI-compatible server.
Put machine-local aliases in `config/aliases.local.toml` (gitignored).

Validate routing with a typed transcript before debugging the microphone:

```bash
just ingest "jenny tell frontend run the tests"
./scripts/run.sh ctl fleet-status
just status
```

## Usage

Say `Jenny, <message>` for a one-shot command. Rename with `[assistant].name`;
when `[wake].phrases` is omitted, `hey <name>`, `okay <name>`, and bare `<name>`
are derived from that setting.

After the wake boundary, an LLM intent plan must name an explicit action and
exact live-catalog workspace/pane IDs. There is no fallback to focus, fuzzy
matching, or automatic single-agent selection. A separate verifier checks the
post-wake transcript and plan before delivery. Direct `prompt` RPC delivery is
disabled.

For longer messages, ask the assistant to begin dictation, continue speaking,
then finish or cancel. Dictation and clarification are planner actions, not
regexes over the transcript.

### Activity UI and Controls

By default the daemon creates a non-focused `voicerdr activity` workspace with
live status and journal history. Press `m` to toggle mute/listen or `q` to close
the viewer. The daemon keeps running when you close the viewer.

`[assistant].avatar = "woman"` (default) shows an ASCII portrait; use `"none"`
to hide it. The viewer button is disabled when busy, shutting down, too narrow,
or already muted. Unknown/error states do not infer listen mode. Set
`[ui] activity_workspace = false` to disable the workspace.

Mouse clicks need SGR reporting (mode 1006) forwarded to the pane; if your
terminal does not forward mouse events, use `m`/`q`. Redirected stdin/stdout
keeps a plain text stream (no application/raw mode). Full history is in
`activity.jsonl` under the state directory - protect it. Pass `--debug` to
`python -m voicerdr.activity` only when opaque payloads are needed.

Explicit `mute`/`listen` choices are stored in `mic_preference.json` and override
`audio.mute_on_start`. Corrupt or interrupted preference state fails closed to
mute. Quit/ensure/rebind freeze the final mode into a private handoff for the
next start; untrusted or incomplete handoffs mean mute. A successful mute waits
for listener closure; if that cannot be proven, the daemon shuts down muted.

### Show available spaces

View space numbers and speakable nicknames (also Herdr action `voicerdr.spaces`):

```bash
just aliases
```

Edit `config/aliases.local.toml` or Herdr's `config/aliases.toml` to customize
nicknames. Toggle `[ui] show_space_numbers` to show/hide #N prefix in Herdr sidebar.

## Model Caches for Offline Use

Python wheels and voice models are separate. After `uv sync`, run this once on
a connected machine:

```bash
./scripts/run.sh bootstrap-models
```

The command reads `audio.stt_model`, `audio.sample_rate`, and `talk.tts_voice`
from the normal config. It uses Moonshine's public model API, Pipecat's public
Kokoro service constructor, NLTK's downloader, and verifies the packaged
Silero model. Re-running is idempotent. Components can be omitted with
`--no-stt`, `--no-tts`, or `--no-tokenizer`.

Default cache locations:

| Asset | Default location | Override |
| --- | --- | --- |
| Moonshine STT | `~/.cache/moonshine_voice/` | `MOONSHINE_VOICE_CACHE` |
| Kokoro ONNX + voices | `~/.cache/pipecat/kokoro-onnx/` | none in the current Pipecat service |
| NLTK `punkt_tab` | usually `~/nltk_data/` | `NLTK_DATA` |
| Silero VAD | inside the installed Pipecat package | none needed |

Preserve those directories when copying an installation to an offline machine,
and keep ownership/read permissions for the user service. The bootstrap does not
provision the external assistant LLM. Test offline readiness by disconnecting the
network, running `bootstrap-models` again (cache hits), then `listen` and `say`.

## Optional systemd Supervision

Detached mode remains the default and requires no systemd. To opt in, invoke
the install action while Herdr is running:

```bash
herdr plugin action invoke voicerdr.service-install
systemctl --user status voicerdr.service
journalctl --user -u voicerdr.service -f
```

This installs `~/.config/systemd/user/voicerdr.service`, enables it for the
user manager, and starts it. The template is rendered with an absolute working
directory and runtime-launcher path; runtime values live in the plugin state
directory's `systemd.env`. Every plugin `ensure` refreshes that file. The unit
uses the same locked Nix runtime as plugin actions when Nix is available, even
when the user manager starts it at login. If Herdr has handed off to a different
socket, ensure cleanly stops the old daemon and starts the service against the
new socket.

Lifecycle commands are idempotent:

```bash
./scripts/run.sh service status
./scripts/run.sh service disable       # quit, disable --now, verify no daemon
./scripts/run.sh service enable        # enable, start, wait for healthy ping
./scripts/run.sh service uninstall     # quit, disable, remove managed files
```

The installer will not overwrite or remove an unmanaged unit named
`voicerdr.service`. Re-run `service install` after moving the checkout or
changing its runtime environment, because `ExecStart` is intentionally absolute.
Do not hand-edit the generated unit; edit `systemd/voicerdr.service.in` and
reinstall it.

## Plugin Lifecycle and Actions

Always stop the daemon before disabling or removing the plugin. The wrapper
does that while Herdr can still inject the correct session paths:

```bash
./scripts/plugin-lifecycle.sh enable
./scripts/plugin-lifecycle.sh disable
./scripts/plugin-lifecycle.sh unlink
./scripts/plugin-lifecycle.sh uninstall
```

To install a tagged release: `herdr plugin install kikijiki/voicerdr --ref vX.Y.Z`.
Herdr runs the manifest's runtime launcher in sync-only mode. Nix-backed installs
receive the locked Python and native runtime; without Nix, `uv` and the documented
native PortAudio requirements must already be installed.

### Common actions

Herdr actions:

```bash
herdr plugin action invoke voicerdr.ensure
herdr plugin action invoke voicerdr.status
herdr plugin action invoke voicerdr.listen
herdr plugin action invoke voicerdr.mute
herdr plugin action invoke voicerdr.quit
herdr plugin action invoke voicerdr.spaces
```

Equivalent checkout commands: `just status`, `just listen`, `just mute`,
`just ingest "…"`, `just aliases`, `just quit`.

### Safe installation

For development:

```bash
herdr plugin link /absolute/path/to/voicerdr
./scripts/plugin-lifecycle.sh enable
```

The startup hook is one-shot. `voicerdr ensure` pings the control socket and
either reuses a daemon bound to the current `HERDR_SOCKET_PATH` or replaces a
stale one. With a voicerdr-managed systemd unit installed, the same command
updates `systemd.env` and starts/restarts the user service instead of spawning a
detached process. Herdr command slots never host the long-running daemon.

## Troubleshooting

Start with these non-destructive checks:

```bash
just doctor
./scripts/run.sh paths
./scripts/run.sh ctl ping
./scripts/run.sh ctl status
./scripts/run.sh service status         # only if systemd was installed
```

### The daemon will not start

- Detached mode logs to the `daemon_log` path under the state directory.
- systemd mode logs to `journalctl --user -u voicerdr.service`.
- `systemctl --user` requires a working user manager/DBus session. Uninstall
  the optional unit and use detached mode in minimal containers or SSH setups
  without one.
- A unit rendered from a moved/deleted checkout must be reinstalled from the
  new path.
- If the control socket is stale, run the safe `quit` action and then `ensure`;
  do not delete arbitrary pid files or kill generic Python processes.

### Herdr events or prompts do not work

- Compare `herdr_socket` from `ctl ping` with `HERDR_SOCKET_PATH`/`paths`.
  Running `ensure` after a Herdr live handoff rebinds the daemon.
- Confirm Herdr is running and `HERDR_BIN_PATH` resolves in the plugin action.
- Confirm the LLM endpoint, model name, and API key in `config.toml`. There is no
  offline routing fallback: planner or verifier failure is intentionally reported
  as "not sent." Check `activity.jsonl` for `interpreting`, `verifying`,
  `verification`, and `withheld` records.
- The direct `prompt` RPC is intentionally disabled. Use wake-addressed typed
  `ingest` for diagnostics; add `--utterance-id` when retry identity must survive
  a delayed response or daemon restart. A transport timeout after prompt commit
  is shown as `delivery_unknown` and must not be retried automatically.
- Run `resolve` or typed `ingest` before investigating microphone capture.

### Activity buttons do not react to clicks

- Use `m` first; it exercises the same direct control-socket path without
  depending on terminal mouse forwarding. `q` only closes the viewer.
- Confirm the pane is receiving standard SGR mouse events. Herdr 0.8.2 supports
  forwarding application mouse mode, but an outer terminal, SSH hop, or nested
  multiplexer may intercept it.
- A dimmed microphone toggle is intentionally disabled because mute is already
  proven in a fail-closed/narrow state, a transition or shutdown is pending, or
  another request is in flight.
- Control failures are shown in `RESULT` and `CONTROL RESULT` history. A failed
  mode response triggers a follow-up Status request; until that succeeds, the UI
  treats the outcome as unknown. Check the control socket and daemon log path
  reported by `./scripts/run.sh paths`; do not delete arbitrary sockets or
  restart unrelated processes.

### Audio fails or selects the wrong device

- Run `just devices`; set `[audio].input_device_index` to a listed input.
- Confirm PortAudio is present (`libportaudio.so` on Linux) and that the user
  can access PipeWire/PulseAudio and the microphone device.
- A systemd user service must run in the same logged-in user session as the
  audio server. Do not use a system service.
- Use headphones. The LocalAudio path does not provide browser-grade acoustic
  echo cancellation, so loud speakers can be transcribed by the microphone.
- First `listen` can take longer while ONNX models load; bootstrap avoids
  network delay but not model initialization time.

### Models try to download unexpectedly

- Run `bootstrap-models` with the same user, `HOME`, config, and cache override
  variables as the daemon.
- Preserve `MOONSHINE_VOICE_CACHE` and `NLTK_DATA` in the systemd environment;
  the installer copies these overrides when it writes `systemd.env`. It also
  captures `PATH`, `LD_LIBRARY_PATH`, and explicit PulseAudio/PipeWire overrides
  so the Nix/native audio runtime remains visible to the user service.
- Kokoro currently follows Pipecat's fixed cache under `~/.cache/pipecat`.

## Limitations

- Linux only; PortAudio/PyAudio is the supported transport.
- One `voicerdr.service` per user. Multiple simultaneous Herdr sessions need
  detached state directories or future instance units.
- Moonshine is currently wired for English transcription.
- The LLM endpoint and LLM weights remain an external operator dependency.
- Wake phrases are an STT text gate, not a hardware-accelerated wake-word
  engine. Push-to-talk and acoustic echo cancellation are not yet implemented.
- Herdr manifests have no disable hook, so raw `herdr plugin disable` cannot
  stop an out-of-band daemon; use the provided lifecycle wrapper.
- SmallWebRTC transport is not supported.
- No marketplace bundle, macOS/Windows service integration, or vendored model
  artifacts are provided.

## Third-Party Software and Models

voicerdr is released under the [MIT License](LICENSE). It builds on the
following projects; their licenses apply to their code and model files, which
are installed or downloaded separately and not redistributed here.

| Component | Use | License |
| --- | --- | --- |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | Audio pipeline, VAD, Kokoro service | BSD-2-Clause |
| [Moonshine Voice](https://github.com/moonshine-ai/moonshine) | Speech-to-text | MIT (code); check each model's license |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) / kokoro-onnx | Text-to-speech | Apache-2.0 (model); see package for code |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection | MIT |
| [Textual](https://github.com/Textualize/textual) | Terminal UI | MIT |
| [NLTK](https://www.nltk.org/) | Sentence tokenizer data | Apache-2.0 |
| [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) / PortAudio | Audio I/O | MIT |

The LLM you point `[llm].base_url` at is an external dependency and is governed
by its own license. Run `uv pip list` or inspect `uv.lock` for the full
transitive dependency set.

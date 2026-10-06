#!/usr/bin/env bash
# Portable entry for Herdr plugin argv. cwd is HERDR_PLUGIN_ROOT when Herdr launches us.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A successful profile refresh reaches this inner invocation. Record its flake
# key only now, so a failed `nix develop` can never bless an older profile.
if [[ "${VOICERDR_NIX_PROFILE_STAMP:-}" == "$ROOT/.voicerdr-nix-runtime.stamp" ]] \
  && [[ -n "${VOICERDR_NIX_PROFILE_KEY:-}" ]]; then
  stamp_tmp="$VOICERDR_NIX_PROFILE_STAMP.$$"
  trap 'rm -f -- "$stamp_tmp"' EXIT
  (umask 077 && printf '%s\n' "$VOICERDR_NIX_PROFILE_KEY" >"$stamp_tmp")
  mv -f -- "$stamp_tmp" "$VOICERDR_NIX_PROFILE_STAMP"
  trap - EXIT
fi
unset VOICERDR_NIX_PROFILE_STAMP VOICERDR_NIX_PROFILE_KEY

# Plugin actions run outside interactive development shells. When this is a
# Nix-backed checkout/install, enter the project's locked runtime once so its
# Python and native shared libraries are also available to detached children.
if [[ "${VOICERDR_NIX_RUNTIME:-}" != "$ROOT" ]] \
  && [[ -f "$ROOT/flake.nix" ]] \
  && command -v nix >/dev/null 2>&1; then
  runtime_profile="$ROOT/.voicerdr-nix-runtime"
  runtime_stamp="$ROOT/.voicerdr-nix-runtime.stamp"
  runtime_key="$(cksum "$ROOT/flake.nix" "$ROOT/flake.lock" | cksum)"
  export VOICERDR_NIX_REEXEC=1
  if [[ -n "$runtime_key" ]] \
    && [[ -e "$runtime_profile" ]] \
    && [[ -f "$runtime_stamp" ]] \
    && [[ "$(<"$runtime_stamp")" == "$runtime_key" ]]; then
    exec nix develop "$runtime_profile" --command \
      env VOICERDR_NIX_RUNTIME="$ROOT" bash "$ROOT/scripts/run.sh" "$@"
  fi
  exec nix develop --profile "$runtime_profile" "$ROOT" --command \
    env \
      VOICERDR_NIX_RUNTIME="$ROOT" \
      VOICERDR_NIX_PROFILE_STAMP="$runtime_stamp" \
      VOICERDR_NIX_PROFILE_KEY="$runtime_key" \
      bash "$ROOT/scripts/run.sh" "$@"
fi
unset VOICERDR_NIX_REEXEC

if ! command -v uv >/dev/null 2>&1; then
  echo "voicerdr: uv not found on PATH" >&2
  exit 127
fi

# Sync quietly on first use; keep plugin actions snappy after that.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --quiet --frozen 2>/dev/null || uv sync --quiet

if [[ "${1:-}" == "--sync-only" ]]; then
  if [[ $# -ne 1 ]]; then
    echo "voicerdr: --sync-only does not accept arguments" >&2
    exit 2
  fi
  exit 0
fi

exec uv run voicerdr "$@"

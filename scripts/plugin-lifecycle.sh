#!/usr/bin/env bash
# Safe wrappers around Herdr plugin enable/disable/install/uninstall operations.
set -euo pipefail

HERDR_BIN="${HERDR_BIN_PATH:-herdr}"
PLUGIN_ID="voicerdr"

usage() {
  echo "usage: $0 {enable|disable|install OWNER/REPO[/SUBDIR]|uninstall|unlink}" >&2
  exit 2
}

disable_daemon() {
  # These actions handle detached daemons too. A failed service cleanup must
  # stop this wrapper before Herdr disables or removes the plugin.
  "$HERDR_BIN" plugin action invoke "$PLUGIN_ID.service-disable"
}

remove_service() {
  "$HERDR_BIN" plugin action invoke "$PLUGIN_ID.service-uninstall"
}

case "${1:-}" in
  enable)
    "$HERDR_BIN" plugin enable "$PLUGIN_ID"
    "$HERDR_BIN" plugin action invoke "$PLUGIN_ID.ensure"
    ;;
  disable)
    disable_daemon
    "$HERDR_BIN" plugin disable "$PLUGIN_ID"
    ;;
  install)
    [[ $# -eq 2 ]] || usage
    "$HERDR_BIN" plugin install "$2"
    "$HERDR_BIN" plugin action invoke "$PLUGIN_ID.ensure"
    ;;
  uninstall)
    remove_service
    "$HERDR_BIN" plugin uninstall "$PLUGIN_ID"
    ;;
  unlink)
    remove_service
    "$HERDR_BIN" plugin unlink "$PLUGIN_ID"
    ;;
  *) usage ;;
esac

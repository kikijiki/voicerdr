import argparse
import json
import os
import sys

from voicerdr import __version__
from voicerdr.config import AppConfig
from voicerdr.control_client import ControlClient, ControlClientError
from voicerdr.daemon import run_daemon
from voicerdr.ensure import print_ensure_result, stop_daemon
from voicerdr.models import bootstrap_models
from voicerdr.operations import (
    OperationsError,
    disable_service,
    enable_service,
    ensure_managed_daemon,
    install_service,
    print_operation_result,
    service_status,
    uninstall_service,
)
from voicerdr.paths import resolve_paths


def _context_workspace_label() -> str | None:
    ctx_raw = os.environ.get("HERDR_PLUGIN_CONTEXT_JSON")
    if not ctx_raw:
        return None
    try:
        ctx = json.loads(ctx_raw)
    except json.JSONDecodeError:
        return None
    label = ctx.get("workspace_label")
    return str(label) if label else None


def _spaces_ui(paths) -> int:
    """Human-readable space directory for the Herdr plugin pane."""
    from voicerdr.config import DEFAULT_ASSISTANT_NAME, AppConfig, load_aliases
    from voicerdr.herdr_client import HerdrClient, HerdrError
    from voicerdr.space_labels import strip_number_prefix

    assistant_name = DEFAULT_ASSISTANT_NAME
    try:
        ensure_managed_daemon(paths)
        client = ControlClient(paths.control_socket, timeout=3.0)
        data = client.call("aliases", {})
        spaces = data.get("spaces") or []
        aliases = data.get("aliases") or {}
        assistant_name = str(data.get("assistant_name") or assistant_name)
    except (ControlClientError, OSError, RuntimeError):
        # Offline fallback without daemon.
        config = AppConfig.load(paths)
        assistant_name = config.assistant_name
        aliases = load_aliases(paths) or config.aliases
        herdr = HerdrClient(paths.herdr_bin, paths.herdr_socket)
        spaces = []
        try:
            for ws in herdr.workspace_list():
                label = str(ws.get("label") or "")
                base = strip_number_prefix(label)
                nicks = sorted(
                    k
                    for k, v in aliases.items()
                    if strip_number_prefix(v).casefold() == base.casefold()
                )
                spaces.append(
                    {
                        "number": ws.get("number"),
                        "label": label,
                        "base": base,
                        "nicknames": nicks,
                    }
                )
        except HerdrError as exc:
            print(f"voicerdr spaces: {exc}", file=sys.stderr)
            return 1

    print("voicerdr spaces — say a number or nickname after the wake phrase")
    print(f"  {assistant_name}, ask <alias|#> <message>")
    print("  (hey/okay optional; tell/ask both work)")
    print()
    if not spaces:
        print("(no Herdr workspaces)")
    for row in spaces:
        num = row.get("number")
        label = row.get("label") or "?"
        base = row.get("base") or label
        nicks = row.get("nicknames") or []
        nick_s = ", ".join(nicks) if nicks else "(add nicknames in aliases.toml)"
        print(f"  #{num:<3} {label:<24} base={base:<16} say: {nick_s}")
    print()
    print("Edit nicknames: config/aliases.local.toml (or HERDR config aliases.toml)")
    print("Toggle #N labels: [ui] show_space_numbers in config.toml")
    print()
    print("Press Enter to close…")
    try:
        input()
    except EOFError:
        pass
    return 0


def _forward_plugin_event(paths) -> int:
    """Parse HERDR_PLUGIN_EVENT_JSON and notify the daemon (best-effort)."""
    raw = os.environ.get("HERDR_PLUGIN_EVENT_JSON")
    if not raw:
        return 0
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    # Envelope shapes vary; dig for pane_id + status.
    blob = json.dumps(envelope)
    pane_id = None
    status = None

    def walk(obj: object) -> None:
        nonlocal pane_id, status
        if isinstance(obj, dict):
            if pane_id is None and obj.get("pane_id"):
                pane_id = str(obj["pane_id"])
            for key in ("agent_status", "status"):
                if status is None and obj.get(key) in (
                    "idle",
                    "working",
                    "blocked",
                    "done",
                    "unknown",
                ):
                    status = str(obj[key])
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(envelope)
    if not pane_id or not status:
        # Still useful for debugging.
        print(json.dumps({"ok": True, "forwarded": False, "keys": list(envelope)[:8]}))
        return 0
    try:
        ensure_managed_daemon(paths)
        client = ControlClient(paths.control_socket, timeout=1.0)
        client.call(
            "notify_agent_event",
            {"pane_id": pane_id, "agent_status": status, "raw": blob[:2000]},
        )
    except Exception as exc:  # noqa: BLE001 — event hooks must not fail loudly
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 0
    print(
        json.dumps(
            {"ok": True, "forwarded": True, "pane_id": pane_id, "status": status}
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voicerdr", description="Configurable Herdr voice assistant"
    )
    parser.add_argument(
        "--version", action="version", version=f"voicerdr {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "ensure", help="Idempotently start the daemon for this Herdr session"
    )

    p_daemon = sub.add_parser("daemon", help="Run the daemon (normally via ensure)")
    p_daemon.add_argument(
        "--foreground",
        action="store_true",
        help="Run in the current process (used by ensure spawn)",
    )

    p_ctl = sub.add_parser("ctl", help="Talk to the running daemon control socket")
    p_ctl.add_argument(
        "action",
        choices=[
            "ping",
            "status",
            "listen",
            "mute",
            "quit",
            "resolve",
            "ingest",
            "aliases",
            "say",
            "fleet-status",
            "talk-policy",
        ],
    )
    p_ctl.add_argument("--space", help="Workspace label/alias")
    p_ctl.add_argument("--agent", help="Agent name or terminal-title phrase")
    p_ctl.add_argument("--text", help="Text for ingest / say")
    p_ctl.add_argument(
        "--utterance-id",
        help="Stable retry identity for typed ingest (same ID is consumed once)",
    )
    p_ctl.add_argument(
        "--mode",
        choices=["silent", "blocked_only", "milestones", "verbose"],
        help="Announcement mode for talk-policy",
    )
    quiet_group = p_ctl.add_mutually_exclusive_group()
    quiet_group.add_argument(
        "--quiet", action="store_true", help="Silence announcements for --space"
    )
    quiet_group.add_argument(
        "--announce", action="store_true", help="Re-enable announcements for --space"
    )

    sub.add_parser(
        "forward-event",
        help="Forward HERDR_PLUGIN_EVENT_JSON (plugin event hook) to the daemon",
    )

    sub.add_parser("paths", help="Print resolved runtime paths as JSON")
    sub.add_parser(
        "spaces-ui",
        help="Print speakable space # / nicknames (Herdr plugin pane)",
    )

    p_models = sub.add_parser(
        "bootstrap-models",
        help="Download/provision local assets now for later offline use",
    )
    p_models.add_argument("--no-stt", action="store_true", help="Skip Moonshine")
    p_models.add_argument("--no-tts", action="store_true", help="Skip Kokoro")
    p_models.add_argument(
        "--no-tokenizer", action="store_true", help="Skip NLTK punkt_tab"
    )

    p_service = sub.add_parser(
        "service", help="Manage optional systemd --user supervision"
    )
    service_sub = p_service.add_subparsers(dest="service_cmd", required=True)
    p_install = service_sub.add_parser("install", help="Install/update the user unit")
    p_install.add_argument(
        "--enable", action="store_true", help="Enable startup at user login"
    )
    p_install.add_argument("--now", action="store_true", help="Ensure it is running")
    service_sub.add_parser("enable", help="Enable and ensure the service")
    service_sub.add_parser("disable", help="Quit the daemon and disable the service")
    service_sub.add_parser("uninstall", help="Quit and remove the managed unit")
    service_sub.add_parser("status", help="Show unit and daemon health")

    args = parser.parse_args(argv)
    paths = resolve_paths()

    if args.cmd == "paths":
        print(
            json.dumps(
                {
                    "plugin_root": str(paths.plugin_root),
                    "config_dir": str(paths.config_dir),
                    "state_dir": str(paths.state_dir),
                    "control_socket": str(paths.control_socket),
                    "herdr_bin": paths.herdr_bin,
                    "herdr_socket": paths.herdr_socket,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "spaces-ui":
        return _spaces_ui(paths)
    if args.cmd == "ensure":
        try:
            result = ensure_managed_daemon(paths)
        except (OperationsError, OSError, RuntimeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
        print_ensure_result(result)
        return 0

    if args.cmd == "bootstrap-models":
        config = AppConfig.load(paths)
        try:
            result = bootstrap_models(
                config,
                tokenizer=not args.no_tokenizer,
                stt=not args.no_stt,
                tts=not args.no_tts,
                progress=lambda message: print(message, file=sys.stderr),
            )
        except Exception as exc:  # noqa: BLE001 — dependency errors need CLI context
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
        print_operation_result(result)
        return 0

    if args.cmd == "service":
        try:
            if args.service_cmd == "install":
                result = install_service(paths, enable=args.enable, start=args.now)
            elif args.service_cmd == "enable":
                result = enable_service(paths)
            elif args.service_cmd == "disable":
                result = disable_service(paths)
            elif args.service_cmd == "uninstall":
                result = uninstall_service(paths)
            elif args.service_cmd == "status":
                result = service_status(paths)
            else:
                return 2
        except (OperationsError, OSError, RuntimeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
        print_operation_result(result)
        return 0 if result.get("ok", True) else 1

    if args.cmd == "daemon":
        # Direct contenders may read existing configuration, but only the
        # singleton winner is allowed to seed or mutate daemon-owned files.
        config = AppConfig.load(paths, seed=False)
        return run_daemon(paths, config)

    if args.cmd == "ctl":
        client = ControlClient(paths.control_socket)
        try:
            if args.action == "ping":
                result = client.ping()
            elif args.action == "status":
                result = client.status()
            elif args.action == "listen":
                result = client.listen()
            elif args.action == "mute":
                result = client.mute()
            elif args.action == "quit":
                result = stop_daemon(paths)
                print(json.dumps({"ok": True, "result": result}, indent=2))
                return 0 if result.get("ok") else 1
            elif args.action == "resolve":
                space = (
                    args.space
                    or os.environ.get("VOICERDR_TEST_SPACE")
                    or _context_workspace_label()
                )
                if not space:
                    print(
                        "ctl resolve requires --space, VOICERDR_TEST_SPACE, "
                        "or plugin workspace context",
                        file=sys.stderr,
                    )
                    return 2
                params = {"space": space}
                if args.agent:
                    params["agent"] = args.agent
                result = client.call("resolve", params)
            elif args.action == "ingest":
                text = args.text or os.environ.get("VOICERDR_TEST_TEXT")
                if not text:
                    print(
                        "ctl ingest requires --text or VOICERDR_TEST_TEXT",
                        file=sys.stderr,
                    )
                    return 2
                # The assistant LLM may need several seconds.
                result = client.ingest_transcript(
                    text, utterance_id=args.utterance_id, timeout=60.0
                )
            elif args.action == "aliases":
                result = client.call("aliases", {})
            elif args.action == "say":
                text = args.text or os.environ.get("VOICERDR_TEST_TEXT")
                if not text:
                    print(
                        "ctl say requires --text or VOICERDR_TEST_TEXT", file=sys.stderr
                    )
                    return 2
                result = client.call("say", {"text": text}, timeout=60.0)
            elif args.action == "fleet-status":
                result = client.call("fleet_status")
            elif args.action == "talk-policy":
                quiet = True if args.quiet else False if args.announce else None
                if not args.mode and not (args.space and quiet is not None):
                    print(
                        "ctl talk-policy requires --mode or --space with "
                        "--quiet/--announce",
                        file=sys.stderr,
                    )
                    return 2
                params = {}
                if args.mode:
                    params["mode"] = args.mode
                if args.space:
                    params["space"] = args.space
                if quiet is not None:
                    params["quiet"] = quiet
                result = client.call("set_talk_policy", params)
            else:
                return 2
        except (ControlClientError, OSError, RuntimeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 1
        print(json.dumps({"ok": True, "result": result}, indent=2))
        return 0

    if args.cmd == "forward-event":
        return _forward_plugin_event(paths)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

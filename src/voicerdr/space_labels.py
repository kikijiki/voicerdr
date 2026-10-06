import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from voicerdr.herdr_client import HerdrClient, HerdrError

log = logging.getLogger(__name__)

_NUM_PREFIX = re.compile(r"^#?\s*(\d+)\s+(.+)$")
_SYNC_LOCK = threading.Lock()


def strip_number_prefix(label: str) -> str:
    text = label.strip()
    # Idempotent: peel repeated "#1 #1 name" prefixes from races.
    for _ in range(4):
        m = _NUM_PREFIX.match(text)
        if not m:
            break
        text = m.group(2).strip()
    return text


def numbered_label(number: int, base: str) -> str:
    base = strip_number_prefix(base)
    return f"#{number} {base}"


def load_base_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v}


def save_base_labels(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp:
            temp.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            Path(temp_name).chmod(0o600)
        except OSError:
            pass
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def sync_space_number_labels(
    herdr: HerdrClient,
    *,
    state_path: Path,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Make Herdr space rows show #N so users know what number to speak.

    When enabled, renames each workspace to ``#{number} {base}`` while remembering
    the base name for aliases. When disabled, restores base labels if we prefixed them.
    """
    with _SYNC_LOCK:
        return _sync_space_number_labels(herdr, state_path=state_path, enabled=enabled)


def _sync_space_number_labels(
    herdr: HerdrClient,
    *,
    state_path: Path,
    enabled: bool,
) -> list[dict[str, Any]]:
    try:
        workspaces = herdr.workspace_list()
    except HerdrError as exc:
        log.debug("space label sync skipped: %s", exc)
        return []

    bases = load_base_labels(state_path)
    changed = False
    directory: list[dict[str, Any]] = []

    for ws in workspaces:
        wid = str(ws.get("workspace_id") or "")
        if not wid:
            continue
        current = str(ws.get("label") or "")
        try:
            number = int(ws.get("number"))
        except (TypeError, ValueError):
            number = None

        observed_base = strip_number_prefix(current) or current
        if wid not in bases:
            bases[wid] = observed_base
            changed = True
        base = bases[wid] or observed_base

        # A label whose base differs from our last recorded value is a user or
        # external rename. Adopt it rather than immediately overwriting it.
        expected_current = (
            numbered_label(number, base) if enabled and number is not None else base
        )
        if current != expected_current and observed_base and observed_base != base:
            base = observed_base
            bases[wid] = base
            changed = True

        if enabled and number is not None:
            desired = numbered_label(number, base)
        else:
            desired = base

        if current != desired:
            try:
                herdr.workspace_rename(wid, desired)
                log.info("space label %s: %r -> %r", wid, current, desired)
                current = desired
            except HerdrError as exc:
                log.warning("rename %s failed: %s", wid, exc)

        directory.append(
            {
                "workspace_id": wid,
                "number": number,
                "label": current,
                "base": base,
            }
        )

    # Drop bases for closed workspaces.
    live = {row["workspace_id"] for row in directory}
    stale = [k for k in bases if k not in live]
    for k in stale:
        del bases[k]
        changed = True
    if changed or stale:
        save_base_labels(state_path, bases)

    return directory

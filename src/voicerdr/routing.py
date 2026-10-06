import re
from dataclasses import dataclass
from typing import Any

from voicerdr.space_labels import strip_number_prefix

# LLM / STT often return spoken slot names ("nine") instead of digits.
_SPOKEN_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass
class ResolveResult:
    ok: bool
    workspace_id: str | None = None
    workspace_label: str | None = None
    target: str | None = None  # agent name or pane id for agent.prompt
    pane_id: str | None = None
    agent_status: str | None = None
    terminal_title: str | None = None
    candidates: list[str] | None = None
    message: str | None = None


def normalize(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch in "-_")


def agent_name(agent: dict[str, Any]) -> str | None:
    """Return Herdr's optional unique agent name, never its implementation kind."""
    value = agent.get("name")
    if value in (None, ""):
        return None
    return str(value)


def _label_keys(label: str) -> list[str]:
    """Match both '#1 frontend' display labels and the bare base name."""
    keys = [normalize(label)]
    base = strip_number_prefix(label)
    nb = normalize(base)
    if nb and nb not in keys:
        keys.append(nb)
    return [k for k in keys if k]


def resolve_workspace(
    query: str,
    workspaces: list[dict[str, Any]],
    aliases: dict[str, str],
) -> ResolveResult:
    q = query.strip()
    if not q:
        return ResolveResult(ok=False, message="empty workspace query")

    # Alias map first (supports multi-word keys like "web app").
    q_key = " ".join(q.lower().split())
    alias_target = aliases.get(q_key) or aliases.get(normalize(q))
    label_query = alias_target or q

    # Herdr workspace slot number: "3", "#3", or spoken "nine".
    if not alias_target:
        num: int | None = None
        num_match = re.fullmatch(r"#?\s*(\d+)", q.strip())
        if num_match:
            num = int(num_match.group(1))
        else:
            num = _SPOKEN_NUMBERS.get(q_key) or _SPOKEN_NUMBERS.get(normalize(q))
        if num is not None:
            for ws in workspaces:
                try:
                    if int(ws.get("number")) == num:
                        return ResolveResult(
                            ok=True,
                            workspace_id=str(ws.get("workspace_id")),
                            workspace_label=str(
                                ws.get("label") or ws.get("workspace_id")
                            ),
                        )
                except (TypeError, ValueError):
                    continue

    scored: list[tuple[int, dict[str, Any]]] = []
    nq = normalize(strip_number_prefix(label_query))
    if not nq:
        return ResolveResult(
            ok=False,
            message=f"no workspace matched {query!r}",
            candidates=[
                str(w.get("label") or w.get("workspace_id")) for w in workspaces
            ],
        )
    for ws in workspaces:
        label = str(ws.get("label") or "")
        for nl in _label_keys(label):
            score = 0
            if nl == nq:
                score = 100
            elif nl.startswith(nq) or nq.startswith(nl):
                score = 80
            elif nq in nl or nl in nq:
                score = 60
            if score:
                scored.append((score, ws))

    # De-dupe same workspace keeping best score.
    best_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for score, ws in scored:
        wid = str(ws.get("workspace_id"))
        prev = best_by_id.get(wid)
        if prev is None or score > prev[0]:
            best_by_id[wid] = (score, ws)
    scored = list(best_by_id.values())

    if not scored:
        for ws in workspaces:
            wt = ws.get("worktree") or {}
            repo = normalize(str(wt.get("repo_name") or ""))
            if repo and (repo == nq or nq in repo or repo in nq):
                scored.append((70, ws))

    if not scored:
        return ResolveResult(
            ok=False,
            message=f"no workspace matched {query!r}",
            candidates=[
                str(w.get("label") or w.get("workspace_id")) for w in workspaces
            ],
        )

    scored.sort(key=lambda x: (-x[0], str(x[1].get("label") or "")))
    best_score = scored[0][0]
    top = [ws for score, ws in scored if score == best_score]
    if len(top) > 1:
        return ResolveResult(
            ok=False,
            message=f"ambiguous workspace {query!r}",
            candidates=[str(w.get("label") or w.get("workspace_id")) for w in top],
        )

    ws = top[0]
    return ResolveResult(
        ok=True,
        workspace_id=str(ws.get("workspace_id")),
        workspace_label=str(ws.get("label") or ws.get("workspace_id")),
    )


def pick_agent_in_workspace(
    workspace_id: str,
    agents: list[dict[str, Any]],
    *,
    prefer_name: str | None = None,
    agent_query: str | None = None,
) -> ResolveResult:
    in_ws = [a for a in agents if str(a.get("workspace_id")) == workspace_id]
    if not in_ws:
        return ResolveResult(
            ok=False,
            workspace_id=workspace_id,
            message=f"no agents in workspace {workspace_id}",
        )

    # A pane id is always a valid prompt target. With exactly one agent pane,
    # routing is unambiguous even if a model copied a workspace token or an
    # "unnamed" catalog placeholder into the optional agent field.
    if len(in_ws) == 1:
        return _agent_result(in_ws[0], workspace_id)

    requested = (agent_query or prefer_name or "").strip()
    if requested:
        matched = _match_agent(requested, in_ws)
        if matched.ok:
            matched.workspace_id = workspace_id
            return matched
        matched.workspace_id = workspace_id
        return matched

    # Multiple panes always need an explicit name/title. A lone named agent is
    # not implicitly primary when another unnamed pane is also active.
    return ResolveResult(
        ok=False,
        workspace_id=workspace_id,
        message="multiple agents in workspace; say which title",
        candidates=[_agent_display(agent) for agent in in_ws],
    )


def _agent_display(agent: dict[str, Any]) -> str:
    assigned_name = agent.get("name")
    implementation = agent.get("agent")
    title = agent.get("terminal_title_stripped")
    if assigned_name:
        return str(assigned_name)
    if implementation and title:
        return f"{implementation} — {title}"
    return str(title or implementation or agent.get("pane_id") or "unnamed agent")


def _match_agent(query: str, agents: list[dict[str, Any]]) -> ResolveResult:
    """Resolve an agent name, pane id, or terminal-title phrase deterministically."""
    nq = normalize(query)
    if not nq:
        return ResolveResult(
            ok=False,
            message="empty agent or title query",
            candidates=[_agent_display(agent) for agent in agents],
        )
    scored: list[tuple[int, dict[str, Any]]] = []
    for agent in agents:
        fields = (
            (agent_name(agent), 100),
            (agent.get("pane_id"), 100),
            (agent.get("terminal_title_stripped"), 95),
        )
        best = 0
        for raw, exact_score in fields:
            value = normalize(str(raw or ""))
            if not value:
                continue
            if value == nq:
                best = max(best, exact_score)
            elif value.startswith(nq) or nq.startswith(value):
                best = max(best, 80)
            elif nq in value or value in nq:
                best = max(best, 70)
        if best:
            scored.append((best, agent))

    candidates = [_agent_display(agent) for agent in agents]
    if not scored:
        return ResolveResult(
            ok=False,
            message=f"no agent or title matched {query!r}",
            candidates=candidates,
        )

    scored.sort(key=lambda item: (-item[0], _agent_display(item[1]).casefold()))
    best_score = scored[0][0]
    top = [agent for score, agent in scored if score == best_score]
    if len(top) != 1:
        return ResolveResult(
            ok=False,
            message=f"ambiguous agent or title {query!r}",
            candidates=[_agent_display(agent) for agent in top],
        )
    return _agent_result(top[0], str(top[0].get("workspace_id") or ""))


def _agent_result(agent: dict[str, Any], workspace_id: str) -> ResolveResult:
    # ``name`` is a unique, user-assigned selector. Current Herdr's ``agent``
    # value describes the implementation and may repeat across panes, so use
    # the pane id as the delivery target when no assigned name is available.
    name = agent.get("name")
    pane_id = agent.get("pane_id")
    target = str(name or pane_id or "")
    if not target:
        return ResolveResult(
            ok=False,
            workspace_id=workspace_id,
            message=f"agent in workspace {workspace_id} has no routable target",
        )
    return ResolveResult(
        ok=True,
        workspace_id=workspace_id,
        target=target,
        pane_id=str(pane_id or "") or None,
        agent_status=str(agent.get("agent_status") or ""),
        terminal_title=str(agent.get("terminal_title_stripped") or ""),
    )


def resolve_route(
    query: str,
    *,
    workspaces: list[dict[str, Any]],
    agents: list[dict[str, Any]],
    aliases: dict[str, str],
    agent_query: str | None = None,
) -> ResolveResult:
    ws = resolve_workspace(query, workspaces, aliases)
    if not ws.ok or not ws.workspace_id:
        return ws
    picked = pick_agent_in_workspace(
        ws.workspace_id,
        agents,
        agent_query=agent_query,
    )
    if not picked.ok:
        picked.workspace_label = ws.workspace_label
        return picked
    picked.workspace_label = ws.workspace_label
    return picked

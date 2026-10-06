import json
import ssl
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from voicerdr.config import (
    DEFAULT_VAD_STOP_SECS,
    MAX_VAD_STOP_SECS,
    MIN_VAD_STOP_SECS,
    AppConfig,
)
from voicerdr.control_protocol import ControlRequest
from voicerdr.ensure import (
    _daemon_healthy,
    _find_voicerdr_pids,
    _read_session,
    _startup_transaction,
    stop_daemon,
)
from voicerdr.intent import (
    IntentTarget,
    IntentValidationError,
    match_wake_phrase,
    plan_from_json,
    strip_wake_phrase,
)
from voicerdr.paths import RuntimePaths
from voicerdr.routing import resolve_route
from voicerdr.secretary import (
    SecretaryClient,
    SecretaryError,
    SecretaryOutputError,
    _assistant_message,
    _plan_schema,
    _strict_json_object,
)
from voicerdr.space_labels import load_base_labels, sync_space_number_labels
from voicerdr.tts import Speaker
from voicerdr.voice import VoiceListener, _ensure_nltk_punkt


def runtime_paths() -> RuntimePaths:
    return RuntimePaths(
        plugin_root=Path("/install/a"),
        config_dir=Path("/config/a"),
        state_dir=Path("/state/a"),
        herdr_bin="herdr",
        herdr_socket="/run/herdr-a.sock",
    )


class AssistantConfigTests(unittest.TestCase):
    @staticmethod
    def paths(root: Path) -> RuntimePaths:
        return RuntimePaths(
            plugin_root=root / "plugin",
            config_dir=root / "config",
            state_dir=root / "state",
            herdr_bin="herdr",
            herdr_socket=None,
        )

    def test_fresh_defaults_are_jenny_with_natural_wake_forms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig.load(self.paths(Path(directory)), seed=False)

        self.assertEqual(config.assistant_name, "Jenny")
        self.assertEqual(config.assistant_avatar, "woman")
        self.assertEqual(config.wake_phrases, ["hey jenny", "okay jenny", "jenny"])

    def test_audio_vad_stop_uses_accuracy_oriented_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig.load(self.paths(Path(directory)), seed=False)

        self.assertEqual(config.vad_stop_secs, DEFAULT_VAD_STOP_SECS)
        self.assertEqual(config.vad_stop_secs, 1.3)

    def test_audio_vad_stop_toml_override_is_parsed_and_safely_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            paths.config_dir.mkdir(parents=True)
            paths.config_toml.write_text("[audio]\nvad_stop_secs = 1.75\n")
            self.assertEqual(
                AppConfig.load(paths, seed=False).vad_stop_secs,
                1.75,
            )

            paths.config_toml.write_text("[audio]\nvad_stop_secs = 0.1\n")
            self.assertEqual(
                AppConfig.load(paths, seed=False).vad_stop_secs,
                MIN_VAD_STOP_SECS,
            )

            paths.config_toml.write_text("[audio]\nvad_stop_secs = 30\n")
            self.assertEqual(
                AppConfig.load(paths, seed=False).vad_stop_secs,
                MAX_VAD_STOP_SECS,
            )

    def test_local_audio_overlay_can_override_vad_stop_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            paths.config_dir.mkdir(parents=True)
            paths.config_toml.write_text("[audio]\nvad_stop_secs = 0.9\n")
            local_config = paths.plugin_root / "config" / "config.local.toml"
            local_config.parent.mkdir(parents=True)
            local_config.write_text("[audio]\nvad_stop_secs = 1.6\n")

            config = AppConfig.load(paths, seed=False)

        self.assertEqual(config.vad_stop_secs, 1.6)

    def test_configured_name_derives_wake_forms_when_phrases_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            paths.config_dir.mkdir(parents=True)
            paths.config_toml.write_text(
                '[assistant]\nname = "Ada Lovelace"\navatar = "none"\n'
            )

            config = AppConfig.load(paths, seed=False)

        self.assertEqual(config.assistant_name, "Ada Lovelace")
        self.assertEqual(config.assistant_avatar, "none")
        self.assertEqual(
            config.wake_phrases,
            ["hey ada lovelace", "okay ada lovelace", "ada lovelace"],
        )

    def test_explicit_legacy_wake_phrases_are_preserved_when_name_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            paths.config_dir.mkdir(parents=True)
            paths.config_toml.write_text(
                '[assistant]\nname = "Ada"\n'
                '[wake]\nphrases = ["hey secretary", "computer"]\n'
            )

            config = AppConfig.load(paths, seed=False)

        self.assertEqual(config.assistant_name, "Ada")
        self.assertEqual(config.wake_phrases, ["hey secretary", "computer"])


class WakePhraseTests(unittest.TestCase):
    def test_configured_tokens_allow_ordinary_asr_punctuation_and_casing(
        self,
    ) -> None:
        cases = (
            ("Hey, Jenny, summarize one.", "summarize one."),
            ("HEY...JENNY: summarize number six.", "summarize number six."),
            ("hey—jEnNy!\tsummarize two.", "summarize two."),
            ("Hey … Jenny — summarize three.", "summarize three."),
        )
        for transcript, expected_remainder in cases:
            with self.subTest(transcript=transcript):
                match = match_wake_phrase(transcript, ["hey jenny"])

                self.assertTrue(match.matched)
                self.assertEqual(match.phrase, "hey jenny")
                self.assertEqual(match.remainder, expected_remainder)
                self.assertEqual(
                    transcript[match.remainder_start :], expected_remainder
                )

    def test_custom_multiword_wake_phrase_uses_the_same_token_rules(self) -> None:
        transcript = "oKaY, Ada; LOVELACE, send the exact résumé."

        match = match_wake_phrase(transcript, ["okay ada lovelace"])

        self.assertTrue(match.matched)
        self.assertEqual(match.phrase, "okay ada lovelace")
        self.assertEqual(match.remainder, "send the exact résumé.")
        self.assertEqual(transcript[match.remainder_start :], match.remainder)

    def test_punctuation_tolerance_does_not_accept_near_matches(self) -> None:
        transcripts = (
            "HeyJenny, summarize one.",
            "Hey, Jennie, summarize one.",
            "Hey, Jennyson, summarize one.",
            "Hey, Jenny's summary is ready.",
            "Hey + Jenny, summarize one.",
            "Well, hey, Jenny, summarize one.",
        )
        for transcript in transcripts:
            with self.subTest(transcript=transcript):
                match = match_wake_phrase(transcript, ["hey jenny"])

                self.assertFalse(match.matched)
                self.assertIsNone(match.phrase)
                self.assertEqual(match.remainder, transcript)
                self.assertEqual(match.remainder_start, 0)


class RoutingTests(unittest.TestCase):
    def test_smoke_routes_and_parses(self) -> None:
        workspaces = [
            {"workspace_id": "w1", "label": "#1 frontend", "number": 1},
            {"workspace_id": "w2", "label": "#2 backend", "number": 2},
        ]
        agents = [
            {
                "workspace_id": "w1",
                "pane_id": "w1:p1",
                "agent_status": "working",
                "terminal_title_stripped": "x",
            },
            {
                "workspace_id": "w2",
                "pane_id": "w2:p1",
                "name": "only",
                "agent_status": "idle",
                "terminal_title_stripped": "y",
            },
        ]
        aliases = {"be": "backend", "fe": "frontend"}
        route = resolve_route(
            "fe", workspaces=workspaces, agents=agents, aliases=aliases
        )
        self.assertTrue(route.ok)
        self.assertEqual(route.target, "w1:p1")
        route = resolve_route(
            "2", workspaces=workspaces, agents=agents, aliases=aliases
        )
        self.assertTrue(route.ok)
        self.assertEqual(route.target, "only")

        woke, rest = strip_wake_phrase(
            "A secretary, tell nine hello", ["hey secretary", "okay secretary"]
        )
        self.assertFalse(woke)
        self.assertEqual(rest, "A secretary, tell nine hello")
        self.assertFalse(strip_wake_phrase("no wake here", ["hey secretary"])[0])
        self.assertFalse(
            strip_wake_phrase(
                "His secretary said tell backend restart it", ["secretary"]
            )[0]
        )
        woke, rest = strip_wake_phrase(
            "Secretary,\talpha\nβeta\u2003gamma", ["secretary"]
        )
        self.assertTrue(woke)
        self.assertEqual(rest, "alpha\nβeta\u2003gamma")

        with self.assertRaisesRegex(SecretaryError, "malformed"):
            _strict_json_object(
                'noise {"action_kind":"no_action"} trailing', purpose="intent plan"
            )

        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            ControlRequest.parse('{"method":"ping","method":"prompt"}')


class LifecycleTests(unittest.TestCase):
    def test_session_json_non_objects_are_normalized(self) -> None:
        for value in (None, [], "pid", 42):
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                paths = RuntimePaths(
                    root, root / "config", root / "state", "herdr", None
                )
                paths.state_dir.mkdir()
                paths.session_file.write_text(json.dumps(value))
                self.assertEqual(_read_session(paths), {})

    def test_stop_rechecks_known_pid_when_final_scan_misses_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            paths.state_dir.mkdir()
            paths.session_file.write_text(json.dumps({"pid": 4321}))
            with (
                patch("voicerdr.ensure._pid_belongs_to_install", return_value=True),
                patch("voicerdr.ensure._pid_alive", return_value=True),
                patch("voicerdr.ensure._find_voicerdr_pids", return_value=[]),
                patch("voicerdr.ensure._kill_tree"),
                patch("voicerdr.ensure.time.sleep"),
                patch("voicerdr.ensure.ControlClient"),
            ):
                result = stop_daemon(paths, wait_secs=0)

            self.assertFalse(result["ok"])
            self.assertEqual(result["remaining"], [4321])
            self.assertTrue(paths.session_file.exists())

    def test_launcher_startup_transactions_serialize_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            first_entered = threading.Event()
            release_first = threading.Event()
            order: list[str] = []

            def first() -> None:
                with _startup_transaction(paths):
                    order.append("first-entered")
                    first_entered.set()
                    release_first.wait(10)
                    order.append("first-leaving")

            def second() -> None:
                first_entered.wait(10)
                with _startup_transaction(paths):
                    order.append("second-entered")

            threads = [threading.Thread(target=first), threading.Thread(target=second)]
            for thread in threads:
                thread.start()
            self.assertTrue(first_entered.wait(10))
            self.assertNotIn("second-entered", order)
            release_first.set()
            for thread in threads:
                thread.join(10)

            self.assertEqual(
                order, ["first-entered", "first-leaving", "second-entered"]
            )

    def test_process_scan_requires_matching_install(self) -> None:
        processes = """101 uv run voicerdr daemon --foreground
202 bash /install/a/scripts/run.sh daemon --foreground
303 uv run voicerdr daemon --foreground
404 python inspect /install/a/src/voicerdr/daemon.py"""
        with (
            patch.object(subprocess, "check_output", return_value=processes),
            patch(
                "voicerdr.ensure._pid_has_runtime_paths",
                side_effect=lambda pid, _paths: pid == 303,
            ),
        ):
            self.assertEqual(_find_voicerdr_pids(runtime_paths()), [202, 303])

    def test_health_check_uses_ping_without_full_status(self) -> None:
        class FakeControlClient:
            def __init__(self, _path: Path, timeout: float) -> None:
                self.timeout = timeout

            def ping(self) -> dict[str, object]:
                return {"pong": True, "herdr_socket": "/run/herdr-a.sock"}

            def status(self) -> dict[str, object]:
                raise AssertionError("health check must not request full status")

        with patch("voicerdr.ensure.ControlClient", FakeControlClient):
            self.assertIsNotNone(_daemon_healthy(runtime_paths()))

    def test_health_check_rejects_missing_session_binding(self) -> None:
        class FakeControlClient:
            def __init__(self, _path: Path, timeout: float) -> None:
                self.timeout = timeout

            def ping(self) -> dict[str, object]:
                return {"pong": True, "herdr_socket": None}

        with patch("voicerdr.ensure.ControlClient", FakeControlClient):
            self.assertIsNone(_daemon_healthy(runtime_paths()))

    def test_health_check_rejects_binding_when_current_session_has_none(self) -> None:
        class FakeControlClient:
            def __init__(self, _path: Path, timeout: float) -> None:
                self.timeout = timeout

            def ping(self) -> dict[str, object]:
                return {"pong": True, "herdr_socket": "/run/old-herdr.sock"}

        paths = runtime_paths()
        unbound_paths = RuntimePaths(
            plugin_root=paths.plugin_root,
            config_dir=paths.config_dir,
            state_dir=paths.state_dir,
            herdr_bin=paths.herdr_bin,
            herdr_socket=None,
        )
        with patch("voicerdr.ensure.ControlClient", FakeControlClient):
            self.assertIsNone(_daemon_healthy(unbound_paths))


class SpaceLabelTests(unittest.TestCase):
    def test_external_rename_updates_persisted_base(self) -> None:
        class FakeHerdr:
            def __init__(self) -> None:
                self.label = "alpha"
                self.renames: list[str] = []

            def workspace_list(self) -> list[dict[str, object]]:
                return [{"workspace_id": "w1", "label": self.label, "number": 1}]

            def workspace_rename(self, _workspace_id: str, label: str) -> None:
                self.renames.append(label)
                self.label = label

        herdr = FakeHerdr()
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "space_bases.json"
            sync_space_number_labels(herdr, state_path=state, enabled=True)
            herdr.label = "#1 user-renamed"
            sync_space_number_labels(herdr, state_path=state, enabled=True)
            self.assertEqual(herdr.label, "#1 user-renamed")
            self.assertEqual(herdr.renames, ["#1 alpha"])
            self.assertEqual(load_base_labels(state), {"w1": "user-renamed"})


class VoiceTests(unittest.TestCase):
    def test_listener_can_stop_from_its_own_thread(self) -> None:
        listener = VoiceListener(on_transcript=lambda _text: None)
        listener._thread = threading.current_thread()
        listener.stop()
        self.assertTrue(listener._stop.is_set())

    def test_nltk_ensure_does_not_change_global_tls_context(self) -> None:
        original = ssl._create_default_https_context
        with (
            patch("nltk.data.find", side_effect=LookupError),
            patch("nltk.download", return_value=True),
        ):
            _ensure_nltk_punkt()
        self.assertIs(ssl._create_default_https_context, original)

    def test_replacement_keeps_cancellation_set_until_worker_switches(self) -> None:
        speaker = Speaker()
        speaker._q.put("old")
        with patch.object(speaker, "start"):
            speaker.speak("new", replace=True)
        self.assertTrue(speaker._cancel.is_set())
        self.assertEqual(speaker._q.get_nowait(), "new")


class SecretaryTests(unittest.TestCase):
    def test_multi_agent_schema_requires_independent_agent_evidence(
        self,
    ) -> None:
        schema = _plan_schema(
            "a" * 64,
            "b" * 64,
            spaces=[
                {
                    "workspace_id": "gamma-id",
                    "agents": [
                        {"pane_id": "gamma:one"},
                        {"pane_id": "gamma:two"},
                    ],
                }
            ],
        )
        decision = schema["schema"]["properties"]["decision"]
        encoded = json.dumps(decision)
        self.assertIn('"enum": ["gamma-id"]', encoded)
        actions = {
            item["properties"]["action_kind"]["const"]: item
            for item in decision["oneOf"]
        }
        self.assertEqual(
            actions["agent_prompt"]["allOf"][0]["then"],
            {"required": ["agent_evidence"]},
        )
        self.assertNotIn("allOf", actions["status"])

    spaces: ClassVar[list[dict[str, object]]] = [
        {
            "number": 9,
            "base": "backend",
            "label": "#9 backend",
            "workspace_id": "w9",
            "nicknames": ["nine"],
            "agents": [
                {
                    "name": None,
                    "pane_id": "w9:p1",
                    "title": "Backend work",
                    "status": "idle",
                }
            ],
        }
    ]

    @staticmethod
    def prompt_json() -> dict[str, object]:
        return {
            "action_kind": "agent_prompt",
            "target": {"workspace_id": "w9", "pane_id": "w9:p1"},
            "message": "run every test",
            "workspace_evidence": {
                "source": "post_wake_content",
                "quote": "nine",
                "catalog_number": 9,
            },
            "evidence": {
                "source": "post_wake_content",
                "quote": "run every test",
            },
            "confidence": 0.99,
            "reason": "explicit target and exact message",
        }

    def test_unsupported_schema_fails_closed_without_transport_downgrade(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        with (
            patch.object(
                client,
                "_post_chat",
                side_effect=SecretaryError("unsupported json_schema response_format"),
            ) as post,
            self.assertRaises(SecretaryError),
        ):
            client._complete([], schema={"name": "x"})
        self.assertEqual(post.call_count, 1)

    def test_reasoning_and_tool_calls_are_rejected_even_with_valid_content(
        self,
    ) -> None:
        for field, value in (
            ("reasoning_content", "I inferred a target"),
            ("analysis", "I inferred a target"),
            ("tool_calls", [{"id": "unsafe"}]),
            ("function_call", {"name": "send"}),
            ("function_result", {"ok": True}),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(SecretaryError, f"unexpected field {field}"),
            ):
                _assistant_message(
                    {"choices": [{"message": {"content": "{}", field: value}}]}
                )

    def test_unexpected_response_fields_are_rejected_by_layer(self) -> None:
        cases = (
            {"thoughts": "hidden", "choices": [{"message": {"content": "{}"}}]},
            {
                "choices": [
                    {
                        "scratchpad": "hidden",
                        "message": {"content": "{}"},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "{}",
                            "commentary": "hidden",
                        }
                    }
                ]
            },
        )
        for envelope in cases:
            with (
                self.subTest(envelope=envelope),
                self.assertRaisesRegex(SecretaryError, "unexpected field"),
            ):
                _assistant_message(envelope)

        self.assertEqual(
            _assistant_message(
                {
                    "id": "response-1",
                    "model": "configured",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "{}",
                            },
                        }
                    ],
                    "usage": {"total_tokens": 1},
                    "timings": {
                        "prompt_n": 10,
                        "prompt_ms": 12.5,
                        "predicted_n": 4,
                    },
                }
            ),
            "{}",
        )

        for envelope in (
            {
                "choices": [{"message": {"content": "{}"}}],
                "timings": {"commentary": "hidden"},
            },
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"completion_tokens_details": {"scratchpad": 2}},
            },
        ):
            with (
                self.subTest(envelope=envelope),
                self.assertRaisesRegex(SecretaryError, "unexpected field"),
            ):
                _assistant_message(envelope)

    def test_response_metadata_rejects_object_types(self) -> None:
        for field in (
            "id",
            "model",
            "object",
            "system_fingerprint",
            "service_tier",
        ):
            with self.subTest(field=field), self.assertRaises(SecretaryError):
                _assistant_message(
                    {
                        field: {"attacker": "controlled"},
                        "choices": [{"message": {"content": "{}"}}],
                    }
                )

    def test_unsafe_finish_reasons_are_rejected(self) -> None:
        for finish_reason in ("tool_calls", "length", "content_filter"):
            with (
                self.subTest(finish_reason=finish_reason),
                self.assertRaisesRegex(SecretaryError, "did not finish with stop"),
            ):
                _assistant_message(
                    {
                        "choices": [
                            {
                                "finish_reason": finish_reason,
                                "message": {"content": "{}"},
                            }
                        ]
                    }
                )

    def test_unknown_empty_fields_are_rejected_at_every_metadata_layer(self) -> None:
        cases = (
            {"unknown": None, "choices": [{"message": {"content": "{}"}}]},
            {"choices": [{"unknown": "", "message": {"content": "{}"}}]},
            {"choices": [{"message": {"content": "{}", "unknown": []}}]},
            {
                "usage": {"unknown": {}},
                "choices": [{"message": {"content": "{}"}}],
            },
            {
                "timings": {"unknown": None},
                "choices": [{"message": {"content": "{}"}}],
            },
            {
                "usage": {"completion_tokens_details": {"unknown": ""}},
                "choices": [{"message": {"content": "{}"}}],
            },
        )
        for envelope in cases:
            with (
                self.subTest(envelope=envelope),
                self.assertRaisesRegex(SecretaryError, "unexpected field unknown"),
            ):
                _assistant_message(envelope)

    def test_actual_llama_cpp_numeric_timings_response_is_accepted(self) -> None:
        self.assertEqual(
            _assistant_message(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1_757_347_200,
                    "model": "local-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "{}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                    "timings": {
                        "prompt_n": 10,
                        "prompt_ms": 12.5,
                        "prompt_per_token_ms": 1.25,
                        "prompt_per_second": 800.0,
                        "predicted_n": 4,
                        "predicted_ms": 8.0,
                        "predicted_per_token_ms": 2.0,
                        "predicted_per_second": 500.0,
                    },
                }
            ),
            "{}",
        )

    def test_response_metadata_rejects_unbounded_integers_as_output_errors(
        self,
    ) -> None:
        enormous = 1 << 100_000
        cases = (
            ("created-positive", {"created": enormous}),
            ("created-negative", {"created": -enormous}),
            ("usage-positive", {"usage": {"total_tokens": enormous}}),
            ("usage-negative", {"usage": {"prompt_tokens": -enormous}}),
            (
                "usage-detail",
                {
                    "usage": {
                        "completion_tokens_details": {"reasoning_tokens": enormous}
                    }
                },
            ),
            ("timing-count", {"timings": {"prompt_n": enormous}}),
            ("timing-positive", {"timings": {"prompt_ms": enormous}}),
            ("timing-negative", {"timings": {"predicted_ms": -enormous}}),
        )
        for label, metadata in cases:
            envelope = {
                "choices": [{"message": {"content": "{}"}}],
                **metadata,
            }
            with self.subTest(label=label), self.assertRaises(SecretaryOutputError):
                _assistant_message(envelope)

        raw_envelope = (
            '{"choices":[{"message":{"content":"{}"}}],'
            '"usage":{"total_tokens":' + "9" * 5_000 + "}}"
        )
        with self.assertRaises(SecretaryOutputError):
            _assistant_message(
                _strict_json_object(raw_envelope, purpose="LLM response envelope")
            )

    def test_response_metadata_rejects_nonfinite_floats_and_booleans(self) -> None:
        cases = (
            {"created": True},
            {"usage": {"total_tokens": True}},
            {"usage": {"prompt_tokens_details": {"cached_tokens": False}}},
            {"timings": {"prompt_n": True}},
            {"timings": {"prompt_ms": True}},
            {"timings": {"prompt_ms": float("nan")}},
            {"timings": {"prompt_per_second": float("inf")}},
            {"timings": {"predicted_per_second": float("-inf")}},
        )
        for metadata in cases:
            envelope = {
                "choices": [{"message": {"content": "{}"}}],
                **metadata,
            }
            with (
                self.subTest(metadata=metadata),
                self.assertRaises(SecretaryOutputError),
            ):
                _assistant_message(envelope)

    def test_duplicate_json_keys_are_rejected_at_every_depth(self) -> None:
        with self.assertRaisesRegex(SecretaryError, "duplicate JSON key"):
            _strict_json_object(
                '{"decision":{"action_kind":"no_action","action_kind":"agent_prompt"}}',
                purpose="intent plan",
            )

    def test_string_null_is_not_coerced_for_irrelevant_fields(self) -> None:
        with self.assertRaisesRegex(IntentValidationError, "extra=.*clarification"):
            plan_from_json(
                {
                    "action_kind": "no_action",
                    "confidence": 0.99,
                    "reason": "not a request",
                    "clarification": "null",
                }
            )

    def test_plan_uses_raw_transcript_state_and_live_catalog(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        utterance_digest = "a" * 64
        catalog_digest = "b" * 64
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "utterance_digest": utterance_digest,
                                "catalog_digest": catalog_digest,
                                "decision": self.prompt_json(),
                            }
                        )
                    }
                }
            ]
        }
        with patch.object(client, "_complete", return_value=response) as complete:
            plan = client.plan(
                "tell nine run every test",
                utterance_id="typed:test",
                raw_transcript="Secretary tell nine run every test",
                activation_phrase="Secretary",
                spaces=self.spaces,
                utterance_digest=utterance_digest,
                catalog_digest=catalog_digest,
                state={"phase": "idle"},
            )

        self.assertEqual(plan.decision.target, IntentTarget("w9", "w9:p1"))
        request = json.loads(complete.call_args.args[0][1]["content"])
        self.assertEqual(
            request["raw_transcript"], "Secretary tell nine run every test"
        )
        self.assertEqual(request["capture_state"]["phase"], "idle")
        self.assertEqual(request["live_workspace_catalog"], self.spaces)
        system = complete.call_args.args[0][0]["content"]
        self.assertIn("Never invent", system)
        self.assertIn("Focus is unavailable", system)
        self.assertIn("post_wake_content", system)
        self.assertIn("ASR morphology and homophone errors", system)
        self.assertIn("catalog_number", system)
        self.assertIn("read-only request", system)
        self.assertIn("OMIT agent_evidence", system)
        self.assertEqual(plan.decision.workspace_evidence.catalog_number, 9)

    def test_catalog_number_is_only_valid_on_workspace_evidence(self) -> None:
        raw = self.prompt_json()
        raw["agent_evidence"] = {
            "source": "post_wake_content",
            "quote": "nine",
            "catalog_number": 9,
        }

        with self.assertRaisesRegex(
            IntentValidationError, "agent_evidence must contain source and quote"
        ):
            plan_from_json(raw)

    def test_catalog_number_rejects_null_boolean_and_nonpositive_values(self) -> None:
        for invalid in (None, True, 0, -1):
            with self.subTest(invalid=invalid):
                raw = self.prompt_json()
                raw["workspace_evidence"]["catalog_number"] = invalid
                with self.assertRaisesRegex(
                    IntentValidationError, "must be a positive integer"
                ):
                    plan_from_json(raw)

    def test_malformed_or_prose_wrapped_json_is_rejected(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        responses = [
            {"choices": [{"message": {"content": "{not json"}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": "reasoning first\n"
                            + json.dumps(self.prompt_json())
                        }
                    }
                ]
            },
        ]
        for response in responses:
            with (
                self.subTest(response=response),
                patch.object(client, "_complete", return_value=response),
                self.assertRaisesRegex(SecretaryError, "malformed intent plan"),
            ):
                client.plan(
                    "tell nine run every test",
                    utterance_id="typed:test",
                    raw_transcript="Secretary tell nine run every test",
                    activation_phrase="Secretary",
                    spaces=self.spaces,
                    utterance_digest="a" * 64,
                    catalog_digest="b" * 64,
                )

    def test_missing_exact_target_is_invalid(self) -> None:
        raw = self.prompt_json()
        raw["target"] = None
        with self.assertRaisesRegex(IntentValidationError, "target must contain"):
            plan_from_json(raw)

    def test_target_evidence_must_be_source_qualified_not_explanatory_text(
        self,
    ) -> None:
        raw = self.prompt_json()
        raw["workspace_evidence"] = "post_wake_content: 'nine'"
        with self.assertRaisesRegex(IntentValidationError, "source and quote"):
            plan_from_json(raw)

    def test_verifier_cannot_retarget_or_repair_approved_prompt(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        proposed = plan_from_json(self.prompt_json())
        changed = {
            "approved": True,
            "utterance_digest": "a" * 64,
            "catalog_digest": "b" * 64,
            "plan_digest": "c" * 64,
            "action_kind": "agent_prompt",
            "target": {"workspace_id": "w9", "pane_id": "w9:p2"},
            "message": "run some tests",
            "checks": {
                "source_exact": True,
                "target_exact": True,
                "action_exact": True,
                "digests_exact": True,
            },
            "reason": "changed it",
            "reason_kind": "approved",
        }
        response = {"choices": [{"message": {"content": json.dumps(changed)}}]}
        with (
            patch.object(client, "_complete", return_value=response),
            self.assertRaisesRegex(SecretaryError, "disagreed"),
        ):
            client.verify_prompt(
                utterance_id="typed:test",
                raw_transcript="Secretary tell nine run every test",
                post_wake_content="tell nine run every test",
                activation_phrase="Secretary",
                proposed=proposed,
                complete_plan=proposed,
                spaces=self.spaces,
                utterance_digest="a" * 64,
                catalog_digest="b" * 64,
                plan_digest="c" * 64,
                source_evidence={"post_wake_content": "tell nine run every test"},
            )

    def test_verifier_receives_exact_post_wake_content_and_boundary(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        proposed = plan_from_json(self.prompt_json())
        raw_result = {
            "approved": True,
            "utterance_digest": "a" * 64,
            "catalog_digest": "b" * 64,
            "plan_digest": "c" * 64,
            "action_kind": "agent_prompt",
            "target": {"workspace_id": "w9", "pane_id": "w9:p1"},
            "message": "run every test",
            "checks": {
                "source_exact": True,
                "target_exact": True,
                "action_exact": True,
                "digests_exact": True,
            },
            "reason": "exact source evidence",
            "reason_kind": "approved",
        }
        response = {"choices": [{"message": {"content": json.dumps(raw_result)}}]}
        with patch.object(client, "_complete", return_value=response) as complete:
            result = client.verify_prompt(
                utterance_id="typed:test",
                raw_transcript="Secretary tell nine run every test",
                post_wake_content="tell nine run every test",
                activation_phrase="Secretary",
                proposed=proposed,
                complete_plan=proposed,
                spaces=self.spaces,
                utterance_digest="a" * 64,
                catalog_digest="b" * 64,
                plan_digest="c" * 64,
                source_evidence={"post_wake_content": "tell nine run every test"},
            )

        self.assertTrue(result.approved)
        request = json.loads(complete.call_args.args[0][1]["content"])
        self.assertEqual(
            request["activation"]["post_wake_content"], "tell nine run every test"
        )
        self.assertEqual(request["activation"]["wake_phrase"], "Secretary")
        self.assertIn("utterance start", request["activation"]["boundary"])
        self.assertEqual(
            request["established_payload_facts"],
            {
                "evidence_source": "post_wake_content",
                "evidence_source_present": True,
                "message_equals_evidence_quote": True,
                "evidence_quote_occurrences": 1,
                "evidence_quote_is_unique_contiguous": True,
                "source_exact": True,
            },
        )
        schema = complete.call_args.kwargs["schema"]["schema"]
        self.assertEqual(
            schema["properties"]["checks"]["properties"]["source_exact"]["const"],
            True,
        )


if __name__ == "__main__":
    unittest.main()

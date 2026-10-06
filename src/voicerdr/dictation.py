import time
from dataclasses import dataclass, field


@dataclass
class DictationBuffer:
    """Accumulate spoken fragments across pauses until a closing phrase."""

    parts: list[str] = field(default_factory=list)
    raw_transcripts: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    active: bool = False

    def start(self, first: str = "", *, raw_transcript: str = "") -> None:
        self.parts = []
        self.raw_transcripts = []
        self.started_at = time.monotonic()
        self.active = True
        if first.strip():
            self.parts.append(first)
        if raw_transcript.strip():
            self.raw_transcripts.append(raw_transcript)

    def append(self, text: str, *, raw_transcript: str = "") -> None:
        if text.strip():
            self.parts.append(text)
        if raw_transcript.strip():
            self.raw_transcripts.append(raw_transcript)

    def clear(self) -> None:
        self.parts.clear()
        self.raw_transcripts.clear()
        self.active = False

    def joined(self) -> str:
        # A newline is an explicit, lossless turn boundary. Do not synthesize a
        # whitespace-collapsed source that cannot prove where bytes came from.
        return "\n".join(self.parts)

    def raw_joined(self) -> str:
        return "\n".join(self.raw_transcripts)

    def record_raw(self, transcript: str) -> None:
        if transcript.strip():
            self.raw_transcripts.append(transcript)

    def age_secs(self) -> float:
        return time.monotonic() - self.started_at

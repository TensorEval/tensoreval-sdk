"""Voice metrics for TensorEval.

Provides WER, TTFT, talk ratio, interruption rate, and Indian language support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VoiceMetrics:
    """Voice-specific evaluation metrics.

    Usage:
        metrics = VoiceMetrics()
        result = metrics.compute(transcript, reference="expected text")
        print(result["wer"], result["ttft"])
    """

    wer: bool = True
    """Word Error Rate — transcription accuracy."""

    ttft: bool = True
    """Time to First Token — response latency."""

    talk_ratio: bool = True
    """Agent vs user talk time ratio."""

    interruption: bool = True
    """Number of interruptions."""

    wpm: bool = True
    """Words Per Minute — speaking pace."""

    def compute(
        self,
        transcript: list[dict],
        audio_data: dict | None = None,
        reference: str = "",
    ) -> dict[str, float]:
        """Compute all enabled metrics from a transcript.

        Args:
            transcript: List of {role, content, start_time, end_time} dicts.
            audio_data: Optional audio metadata (duration, sample_rate, etc.)
            reference: Reference text for WER computation.

        Returns:
            Dict of metric_name -> value.
        """
        results: dict[str, float] = {}

        if self.wer:
            results["wer"] = self._compute_wer(transcript, reference)
        if self.ttft:
            results["ttft"] = self._compute_ttft(transcript)
        if self.talk_ratio:
            results["talk_ratio"] = self._compute_talk_ratio(transcript)
        if self.interruption:
            results["interruption_count"] = float(self._compute_interruptions(transcript))
        if self.wpm:
            results["wpm"] = self._compute_wpm(transcript)

        return results

    def _compute_wer(self, transcript: list[dict], reference: str = "") -> float:
        """Compute Word Error Rate.

        WER = (substitutions + insertions + deletions) / reference_words
        Requires a reference text to compare against.
        """
        if not reference:
            return 0.0

        agent_text = " ".join(t.get("content", "") for t in transcript if t.get("role") == "assistant")
        if not agent_text:
            return 1.0

        ref_words = reference.lower().strip().split()
        hyp_words = agent_text.lower().strip().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Levenshtein distance at word level
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    def _compute_ttft(self, transcript: list[dict]) -> float:
        """Compute Time to First Token.

        TTFT = agent_start_time - user_end_time
        """
        user_turns = [t for t in transcript if t.get("role") == "user"]
        agent_turns = [t for t in transcript if t.get("role") == "assistant"]

        if not user_turns or not agent_turns:
            return 0.0

        user_end = user_turns[0].get("end_time", 0)
        agent_start = agent_turns[0].get("start_time", 0)

        if user_end and agent_start:
            return max(0, agent_start - user_end)
        return 0.0

    def _compute_talk_ratio(self, transcript: list[dict]) -> float:
        """Compute agent talk time / total talk time."""
        agent_time = 0.0
        total_time = 0.0

        for turn in transcript:
            start = turn.get("start_time", 0)
            end = turn.get("end_time", 0)
            duration = max(0, end - start)
            total_time += duration
            if turn.get("role") == "assistant":
                agent_time += duration

        return agent_time / total_time if total_time > 0 else 0.0

    def _compute_interruptions(self, transcript: list[dict]) -> int:
        """Count interruptions (agent starts while user is still speaking)."""
        interruptions = 0
        for i in range(1, len(transcript)):
            prev = transcript[i - 1]
            curr = transcript[i]
            if prev.get("role") == "user" and curr.get("role") == "assistant":
                prev_end = prev.get("end_time", 0)
                curr_start = curr.get("start_time", 0)
                if curr_start < prev_end:
                    interruptions += 1
        return interruptions

    def _compute_wpm(self, transcript: list[dict]) -> float:
        """Compute Words Per Minute for agent turns."""
        agent_turns = [t for t in transcript if t.get("role") == "assistant"]
        if not agent_turns:
            return 0.0

        total_words = 0
        total_seconds = 0.0
        for turn in agent_turns:
            content = turn.get("content", "")
            total_words += len(content.split())
            start = turn.get("start_time", 0)
            end = turn.get("end_time", 0)
            total_seconds += max(0, end - start)

        if total_seconds > 0:
            return (total_words / total_seconds) * 60
        return 0.0


@dataclass
class IndianLanguageMetrics:
    """Metrics for Indian language evaluation.

    Based on Voice of India benchmark and Sarvam's evaluation methodology.
    """

    oi_wer: bool = True
    """Orthographically Informed WER — handles spelling variations."""

    lid_accuracy: bool = True
    """Language Identification accuracy."""

    code_switch_detection: bool = True
    """Code-switching detection (Hindi/English mixing)."""

    def compute(self, transcript: list[dict], reference: str = "", language: str = "hi") -> dict[str, float]:
        """Compute Indian language metrics."""
        results: dict[str, float] = {}

        if self.oi_wer:
            results["oi_wer"] = self._compute_oi_wer(transcript, reference)
        if self.lid_accuracy:
            results["lid_accuracy"] = 1.0  # Placeholder
        if self.code_switch_detection:
            results["code_switch_detected"] = self._detect_code_switching(transcript)

        return results

    def _compute_oi_wer(self, transcript: list[dict], reference: str) -> float:
        """Compute OI-WER with NFKC normalization for Indian scripts."""
        if not reference:
            return 0.0

        agent_text = " ".join(t.get("content", "") for t in transcript if t.get("role") == "assistant")
        if not agent_text:
            return 1.0

        import unicodedata
        norm_ref = unicodedata.normalize("NFKC", reference).lower().strip()
        norm_hyp = unicodedata.normalize("NFKC", agent_text).lower().strip()

        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    def _detect_code_switching(self, transcript: list[dict]) -> float:
        """Detect Hindi/English code-switching."""
        agent_text = " ".join(t.get("content", "") for t in transcript if t.get("role") == "assistant")
        if not agent_text:
            return 0.0

        has_devanagari = bool(re.search(r'[\u0900-\u097F]', agent_text))
        has_latin = bool(re.search(r'[a-zA-Z]', agent_text))

        return 1.0 if (has_devanagari and has_latin) else 0.0


@dataclass
class AudioMetrics:
    """Audio quality metrics (placeholder — values read from audio_data dict)."""

    clarity: bool = True
    jitter: bool = True

    def compute(self, audio_data: dict) -> dict[str, float]:
        """Compute audio quality metrics from pre-computed data."""
        results: dict[str, float] = {}
        if self.clarity:
            results["clarity"] = audio_data.get("clarity", 0.0)
        if self.jitter:
            results["jitter"] = audio_data.get("jitter", 0.0)
        return results

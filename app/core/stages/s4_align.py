"""
Stage 4 — Transcript-Speaker Alignment

Matches each Whisper segment to its most likely speaker via timestamp overlap.
Writes to context: 'dialogue' (List[DialogueTurn])
"""
from __future__ import annotations

from typing import Optional

from app.core.schemas import DialogueTurn, StageStatus
from app.core.stages.base import PipelineContext, PipelineStage


def _align(segments, diarization_turns) -> list[DialogueTurn]:
    """Align Whisper segments with speaker turns via maximum timestamp overlap."""
    dialogue = []
    for seg in segments:
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0

        for turn in diarization_turns:
            overlap = max(0.0, min(seg.end, turn["end"]) - max(seg.start, turn["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn["speaker"]

        # Nearest-neighbor fallback when no overlap found
        if max_overlap == 0.0 and diarization_turns:
            closest = min(
                diarization_turns,
                key=lambda t: min(abs(seg.start - t["end"]), abs(seg.end - t["start"])),
            )
            best_speaker = closest["speaker"]

        dialogue.append(
            DialogueTurn(speaker=best_speaker, start=seg.start, end=seg.end, text=seg.text)
        )
    return dialogue


class AlignmentStage(PipelineStage):
    name = "alignment"
    description = "Map each transcript segment to its speaker"
    optional = True

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        segments = context.get("segments", [])
        diarization_turns = context.get("diarization_turns", [])

        dialogue = _align(segments, diarization_turns)
        context["dialogue"] = dialogue

        # Count unique speakers in result
        unique_speakers = len({d.speaker for d in dialogue})
        return StageStatus.SUCCESS, f"{len(dialogue)} turns, {unique_speakers} unique speakers"

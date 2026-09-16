"""Speaker-conditioned audio evidence sidecar for M.E.O.W.

This package deliberately does not redefine v2.3 identity, semantic, dedup, or
split authority.  It produces traceable audio evidence and role-preserving
training views linked to those authorities by stable identifiers.
"""

from .contracts import (
    AUDIO_EVIDENCE_SCHEMA_VERSION,
    ENROLLMENT_BANK_SCHEMA_VERSION,
    MATERIALIZATION_VERSION,
    PIPELINE_VERSION,
    Interval,
    Timebase,
)

__all__ = [
    "AUDIO_EVIDENCE_SCHEMA_VERSION",
    "ENROLLMENT_BANK_SCHEMA_VERSION",
    "MATERIALIZATION_VERSION",
    "PIPELINE_VERSION",
    "Interval",
    "Timebase",
]

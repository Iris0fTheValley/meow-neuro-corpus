from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .contracts import ContractError, ENROLLMENT_BANK_SCHEMA_VERSION, Interval, canonical_sha256


class VerificationStatus(str, Enum):
    CONFIRMED_TARGET = "CONFIRMED_TARGET"
    REJECTED = "REJECTED"
    UNVERIFIED = "UNVERIFIED"


class VerificationMethod(str, Enum):
    HUMAN_GOLD = "HUMAN_GOLD"
    EXISTING_IDENTITY_PLUS_INDEPENDENT = "EXISTING_IDENTITY_PLUS_INDEPENDENT"
    MULTI_EVIDENCE_CONSENSUS = "MULTI_EVIDENCE_CONSENSUS"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


class VerificationDecision(str, Enum):
    SUPPORTS_CONFIRMATION = "SUPPORTS_CONFIRMATION"
    REJECTS_CONFIRMATION = "REJECTS_CONFIRMATION"
    INCONCLUSIVE = "INCONCLUSIVE"


FORBIDDEN_BOOTSTRAP_DOMAINS = {
    "TARGET_ACTIVITY", "PVAD", "NEW_DIARIZATION", "TSE", "NEW_ASR",
    "SEMANTIC_CONTENT", "PERSONA_STYLE", "ENROLLMENT_SIMILARITY",
}


@dataclass(frozen=True)
class VerificationEvidence:
    evidence_id: str
    authority_domain: str
    source_artifact: str
    decision: VerificationDecision
    revision: str

    def __post_init__(self) -> None:
        if not all((self.evidence_id, self.authority_domain, self.source_artifact, self.decision, self.revision)):
            raise ContractError("verification evidence must be fully traceable")
        if not isinstance(self.decision, VerificationDecision):
            raise ContractError("verification evidence decision must use VerificationDecision")
        if self.authority_domain.upper() in FORBIDDEN_BOOTSTRAP_DOMAINS:
            raise ContractError("circular/new-pipeline evidence cannot establish enrollment trust")


@dataclass(frozen=True)
class EnrollmentReference:
    enrollment_id: str
    target_identity: str
    source_recording: str
    source_interval: Interval
    source_provenance: Dict[str, Any]
    verification_method: VerificationMethod
    verification_evidence: List[VerificationEvidence]
    verification_status: VerificationStatus
    audio_quality: Dict[str, Any]
    overlap_status: str
    duration: float
    embedding_backend: str
    embedding_revision: str
    checksum: str
    raw_audio_uri: str
    embedding_uri: Optional[str] = None
    target_subtype: Optional[str] = None
    family_group: str = "NEURO_FAMILY"
    synthetic_test_only: bool = False

    def __post_init__(self) -> None:
        if not all((self.enrollment_id, self.target_identity, self.source_recording, self.source_provenance, self.audio_quality, self.embedding_backend, self.embedding_revision, self.checksum, self.raw_audio_uri)):
            raise ContractError("enrollment provenance is incomplete")
        if abs(self.duration - (self.source_interval.end - self.source_interval.start)) > 1e-6:
            raise ContractError("enrollment duration does not match exact source interval")
        if self.overlap_status != "NO_OVERLAP_CONFIRMED" and self.verification_status == VerificationStatus.CONFIRMED_TARGET:
            raise ContractError("confirmed enrollment cannot contain unresolved overlap")
        if len(self.checksum) != 64:
            raise ContractError("enrollment audio checksum must be SHA-256")
        if self.verification_status == VerificationStatus.CONFIRMED_TARGET:
            self._validate_confirmation()

    def _validate_confirmation(self) -> None:
        if any(item.decision != VerificationDecision.SUPPORTS_CONFIRMATION for item in self.verification_evidence):
            raise ContractError("negative or inconclusive evidence cannot confirm enrollment")
        if self.verification_method == VerificationMethod.SYNTHETIC_FIXTURE:
            if not self.synthetic_test_only or self.embedding_uri:
                raise ContractError("synthetic enrollment is architecture-test-only and cannot publish an embedding")
            return
        if self.synthetic_test_only:
            raise ContractError("real confirmed enrollment cannot be marked synthetic")
        domains = {item.authority_domain for item in self.verification_evidence}
        if self.verification_method == VerificationMethod.HUMAN_GOLD:
            if not domains & {"HUMAN_GOLD", "KNOWN_OFFICIAL_SINGLE_SPEAKER"}:
                raise ContractError("gold enrollment requires gold/known-source evidence")
        elif self.verification_method in {VerificationMethod.EXISTING_IDENTITY_PLUS_INDEPENDENT, VerificationMethod.MULTI_EVIDENCE_CONSENSUS}:
            required = {"EXISTING_IDENTITY_AUTHORITY", "SOURCE_PROVENANCE", "CLEAN_SINGLE_SPEAKER"}
            if not required.issubset(domains):
                raise ContractError("confirmed enrollment lacks independent identity, provenance, or clean-speaker evidence")
        else:
            raise ContractError("unsupported confirmation method")
        flags = self.audio_quality
        if not flags.get("speaker_boundary_clear") or flags.get("identity_conflict") or flags.get("hard_negative_collision"):
            raise ContractError("confirmed enrollment fails strict identity-quality conditions")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["source_interval"] = self.source_interval.to_dict()
        value["verification_method"] = self.verification_method.value
        value["verification_status"] = self.verification_status.value
        return value


@dataclass
class EnrollmentBank:
    bank_id: str
    revision: str
    target_identity: str = "NEURO_FAMILY"
    schema_version: str = ENROLLMENT_BANK_SCHEMA_VERSION
    references: Dict[str, EnrollmentReference] = field(default_factory=dict)

    def add(self, reference: EnrollmentReference) -> None:
        if reference.verification_status != VerificationStatus.CONFIRMED_TARGET:
            raise ContractError("NO_UNVERIFIED_ENROLLMENT: only confirmed target references may enter the bank")
        if reference.target_identity not in {"NEURO", "EVIL_NEURO", "NEURO_FAMILY"}:
            raise ContractError("guest/unknown identity cannot enter the Neuro enrollment bank")
        if reference.enrollment_id in self.references:
            raise ContractError("duplicate enrollment id")
        self.references[reference.enrollment_id] = reference

    def confirmed(self, enrollment_id: str, *, allow_synthetic: bool = False) -> EnrollmentReference:
        reference = self.references.get(enrollment_id)
        if reference is None or reference.verification_status != VerificationStatus.CONFIRMED_TARGET:
            raise ContractError("enrollment is absent or not confirmed")
        if reference.synthetic_test_only and not allow_synthetic:
            raise ContractError("synthetic enrollment cannot be used in production")
        return reference

    def produce_embedding(self, enrollment_id: str, adapter: Any) -> Dict[str, Any]:
        """Gate formal embedding production through the confirmed-only bank."""
        reference = self.confirmed(enrollment_id, allow_synthetic=False)
        output = adapter.embed(reference)
        if not isinstance(output, dict) or not output.get("embedding_uri") or not output.get("embedding_checksum"):
            raise ContractError("embedding producer returned incomplete provenance")
        return {
            "enrollment_id": reference.enrollment_id,
            "source_audio_checksum": reference.checksum,
            "source_recording": reference.source_recording,
            "source_interval": reference.source_interval.to_dict(),
            "verification_evidence": [asdict(item) for item in reference.verification_evidence],
            "embedding_backend": adapter.provenance.backend,
            "embedding_revision": adapter.provenance.revision,
            "embedding_uri": output["embedding_uri"],
            "embedding_checksum": output["embedding_checksum"],
        }

    def manifest(self) -> Dict[str, Any]:
        refs = [self.references[key].to_dict() for key in sorted(self.references)]
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "revision": self.revision,
            "target_identity": self.target_identity,
            "confirmed_only": True,
            "reference_count": len(refs),
            "references_sha256": canonical_sha256(refs),
        }

    def write(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        references = [self.references[key].to_dict() for key in sorted(self.references)]
        (root / "manifest.json").write_text(json.dumps(self.manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "references.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in references), encoding="utf-8")

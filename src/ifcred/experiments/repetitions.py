"""Order-independent random-seed schedules for repeated experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


STANDARD_SEED_ROLES = (
    "split",
    "tuning",
    "model",
    "injection",
    "comparator",
    "bootstrap",
)
RECOMMENDED_REPETITIONS = 30


@dataclass(frozen=True)
class ExperimentRepetition:
    """One repetition with stable, independently named random streams."""

    repetition: int
    root_seed: int

    def __post_init__(self) -> None:
        if self.repetition < 0:
            raise ValueError("repetition must be non-negative")
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")

    def seed_for(self, role: str, stream: str | None = None) -> int:
        """Derive a uint32 seed independent of call and condition ordering."""

        if not role.strip() or (stream is not None and not stream.strip()):
            raise ValueError("seed role and optional stream must be non-empty")
        namespace = role if stream is None else f"{role}:{stream}"
        material = (
            f"ifcred-seed-v1|{self.root_seed}|{self.repetition}|{namespace}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")

    def standard_seeds(self) -> dict[str, int]:
        return {role: self.seed_for(role) for role in STANDARD_SEED_ROLES}

    def injection_seed(self, injection_type: str) -> int:
        """Pair severity levels by sharing a stream within an injection type."""

        return self.seed_for("injection", injection_type)

    def comparator_seed(self, framework_name: str) -> int:
        """Give each prior framework a stable stream within the repetition."""

        return self.seed_for("comparator", framework_name)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "repetition": self.repetition,
            "root_seed": self.root_seed,
            "derivation": "sha256 named streams v1; first uint32",
            "standard_seeds": self.standard_seeds(),
        }


@dataclass(frozen=True)
class RepetitionPlan:
    """A declared number of paired repetitions from one recorded root seed."""

    root_seed: int
    n_repetitions: int = RECOMMENDED_REPETITIONS

    def __post_init__(self) -> None:
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if self.n_repetitions < 2:
            raise ValueError("a repeated experiment requires at least two repetitions")

    @property
    def repetitions(self) -> tuple[ExperimentRepetition, ...]:
        return tuple(
            ExperimentRepetition(repetition=index, root_seed=self.root_seed)
            for index in range(self.n_repetitions)
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "n_repetitions": self.n_repetitions,
            "recommended_repetitions": RECOMMENDED_REPETITIONS,
            "repetitions": [run.to_manifest() for run in self.repetitions],
        }

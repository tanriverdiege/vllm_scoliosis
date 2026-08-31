"""File-backed configuration for the VLM wrappers.

Every argument `VLM.__init__` understands lives in `VLMConfig`, and a YAML
file is the canonical way to fill it in. Holding the knobs in one serialisable
object means a run can be reproduced from the file that produced it -- see
`VLMConfig.to_yaml`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Backends `device` may name. A trailing index ("cuda:1") is allowed, so only
# the part before the colon is matched.
_DEVICE_BACKENDS = ("cpu", "cuda", "mps")

# Flags validated as a group; bool is a subclass of int, so an explicit
# isinstance check is the only way to reject `output_scores: 1`.
_BOOL_FIELDS = (
    "do_sample",
    "return_dict_in_generate",
    "output_scores",
    "output_logits",
)


@dataclass(frozen=True)
class VLMConfig:
    """Every argument the VLM constructor takes.

    Frozen on purpose: the model is loaded from these values once, so a later
    mutation would describe a run that never happened.

    Attributes:
        model_name: A HuggingFace model id.
        device: Device to place the model on, or None to auto-detect.
        do_sample: True samples from the distribution; False takes the argmax
            at every step (deterministic).
        max_new_tokens: Cap on generated tokens.
        return_dict_in_generate: Return a structured output from `generate`
            instead of a bare tensor. Forced on automatically when scores or
            logits are requested, since they are only reachable through the
            structured form.
        output_scores: Collect post-processing distributions per step.
        output_logits: Collect raw logits per step.
    """

    model_name: str
    device: str | None = None
    do_sample: bool = False
    max_new_tokens: int = 128
    return_dict_in_generate: bool = False
    output_scores: bool = False
    output_logits: bool = False

    def __post_init__(self) -> None:
        """Validate the fields and apply the scores/logits implication.

        Raises:
            TypeError: If a field holds a value of the wrong type.
            ValueError: If `model_name` is blank, `device` names an unknown
                backend, or `max_new_tokens` is not positive.
        """
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError(
                f"model_name must be a non-empty HuggingFace model id, got "
                f"{self.model_name!r}"
            )

        if self.device is not None:
            if not isinstance(self.device, str):
                raise TypeError(
                    f"device must be a string or null, got {type(self.device).__name__}"
                )
            backend = self.device.split(":", 1)[0]
            if backend not in _DEVICE_BACKENDS:
                raise ValueError(
                    f"device {self.device!r} names an unknown backend "
                    f"{backend!r}; expected one of {', '.join(_DEVICE_BACKENDS)}, "
                    "optionally with an index such as 'cuda:1', or null to "
                    "auto-detect"
                )

        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens, int
        ):
            raise TypeError(
                f"max_new_tokens must be an int, got "
                f"{type(self.max_new_tokens).__name__}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be at least 1, got {self.max_new_tokens}"
            )

        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

        # Asking for scores/logits without the dict form silently discards
        # them, so treat the request as implying it. Frozen dataclasses block
        # plain assignment; object.__setattr__ is the documented way in.
        if self.output_scores or self.output_logits:
            object.__setattr__(self, "return_dict_in_generate", True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VLMConfig:
        """Build a config from a YAML file.

        Unknown keys are rejected rather than ignored: in a config file a
        typo'd key is a setting that silently did not apply.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.

        Raises:
            TypeError: If the file's top level is not a mapping.
            ValueError: If the file is empty or holds keys that are not
                fields of `VLMConfig`.
        """
        path = Path(path)
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raise ValueError(f"{path} is empty; it must define at least model_name")
        if not isinstance(raw, dict):
            raise TypeError(
                f"{path} must hold a YAML mapping of settings, got {type(raw).__name__}"
            )

        known = {field.name for field in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: "
                f"{', '.join(sorted(str(key) for key in unknown))}; "
                f"expected any of {', '.join(sorted(known))}"
            )
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Write the config back out, so a run can record what produced it.

        Field order is preserved rather than sorted, to keep the written file
        readable next to the hand-written one.

        Args:
            path: Destination file; overwritten if it exists.
        """
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

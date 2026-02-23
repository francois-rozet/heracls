"""Types and protocols."""

from typing import Any, ClassVar, Protocol


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

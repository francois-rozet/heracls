"""CLI utils and parsing."""

__all__ = [
    "ArgumentParser",
    "choice",
]

from functools import partial
from simple_parsing import ArgumentParser, SimpleHelpFormatter, subgroups
from typing import Callable, Dict, Optional, TypeVar

T = TypeVar("T")

ArgumentParser.__init__.__kwdefaults__["formatter_class"] = partial(
    SimpleHelpFormatter,
    max_help_position=64,
    width=96,
)


def choice(
    options: Dict[str, T],
    default: Optional[str] = None,
    default_factory: Optional[Callable[[], T]] = None,
) -> T:
    """Create a field that gives a choice between named options.

    Should be used in combination with :func:`ArgumentParser.add_arguments`.
    """
    if default is None and default_factory is None:
        return subgroups(options)
    elif default is None:
        return subgroups(options, default_factory=default_factory)
    elif default_factory is None:
        return subgroups(options, default=default)
    else:
        return subgroups(options, default=default, default_factory=default_factory)

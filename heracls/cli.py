"""CLI utils and parsing."""

from __future__ import annotations

__all__ = [
    "ArgumentParser",
    "field",
]

import argparse
import copy
import dataclasses
import json

from collections.abc import Callable
from dataclasses import MISSING, dataclass
from functools import partial
from numbers import Number
from typing import (
    Any,
    TypeVar,
    get_args,
)
from unittest.mock import patch

from .core import from_dict
from .typing import Dataclass, get_origin, is_dataclass_type, is_literal, is_union, type_repr

T = TypeVar("T")
DC = TypeVar("DC", bound=Dataclass)
HELP = "<HELP>"


def boolean(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "on"}


def string_parser(t: type[T]) -> Callable[[str], T]:
    origin = get_origin(t)

    if issubclass(origin, bool):
        return boolean
    elif issubclass(origin, (str, Number)):
        parser = lambda s: t(s)
    elif issubclass(origin, (dict, list, tuple)):
        parser = lambda s: json.loads(s)
    else:
        parser = lambda s: s

    parser.__name__ = type_repr(t)

    return parser


def field(
    *,
    choices: dict[str, type[Dataclass] | Callable[[], Any]],
    **kwargs,  # noqa: ANN003
) -> Any:  # noqa: ANN401
    """Create a dataclass field that declares named choices for the parser."""

    assert choices

    metadata = kwargs.setdefault("metadata", {})
    metadata.update(heracls_choices=choices)

    return dataclasses.field(**kwargs)


class ChoiceAction(argparse.Action):
    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        table: dict[str, type[Dataclass] | Callable[[], Any]],
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(option_strings, dest, **kwargs)
        self.table = table
        self.dest_value = dest.removesuffix(":choice")

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        choice: str,
        flag: str | None = None,
    ) -> None:
        value = self.table[choice]

        if is_dataclass_type(value):
            if hasattr(namespace, self.dest_value):
                delattr(namespace, self.dest_value)
        else:
            setattr(namespace, self.dest_value, value())

        setattr(namespace, self.dest, choice)


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter,
):
    def _get_default_metavar_for_optional(self, action: argparse.Action) -> str:
        return getattr(action.type, "__name__", None)

    def _get_help_string(self, *args, **kwargs) -> str | None:  # noqa: ANN002, ANN003
        help = super()._get_help_string(*args, **kwargs)
        if help is not None:
            help = help.replace(HELP, "")
        return help


@dataclass
class DataclassSpec:
    data_cls: type[Dataclass]
    keys: set[str]
    choices: dict[str, dict[str, type[Dataclass] | Callable[[], Any]]]


class ArgumentParser(argparse.ArgumentParser):
    """Simple dataclass-aware argument parser."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        kwargs.setdefault(
            "formatter_class",
            partial(
                HelpFormatter,
                max_help_position=64,
                width=96,
            ),
        )

        super().__init__(*args, **kwargs)

        self.heracls_specs: dict[str, DataclassSpec] = {}

    def add_arguments(
        self,
        data_cls: type[Dataclass],
        dest: str = "config",
        root: bool = False,
    ) -> None:
        assert is_dataclass_type(data_cls)
        assert dest not in self.heracls_specs

        spec = DataclassSpec(
            data_cls=data_cls,
            keys=set(),
            choices={},
        )

        self.heracls_specs[dest] = spec
        self._register_dataclass(spec, data_cls, () if root else (dest,))

    def _register_dataclass(
        self,
        spec: DataclassSpec,
        data_cls: type[Dataclass],
        path: tuple[str, ...] = (),
    ) -> None:
        if path:
            group = self.add_argument_group(".".join(path))
        else:
            group = self.add_argument_group()

        for f in dataclasses.fields(data_cls):
            f_path = (*path, f.name)
            f_dest = ".".join(f_path)
            f_flag = f"--{f_dest}"

            spec.keys.add(f_dest)

            kwargs = {}

            if "heracls_choices" in f.metadata:
                spec.choices[f_dest] = table = f.metadata["heracls_choices"]

                kwargs["dest"] = f"{f_dest}:choice"
                kwargs["default"] = next(iter(table.keys()))
                kwargs["help"] = HELP

                group.add_argument(
                    f_flag,
                    choices=tuple(table.keys()),
                    action=ChoiceAction,
                    table=table,
                    **kwargs,
                )
            elif is_dataclass_type(f.type):
                self._register_dataclass(spec, f.type, f_path)
            else:
                kwargs["dest"] = f_dest

                if f.default is MISSING and f.default_factory is MISSING:
                    kwargs["default"] = argparse.SUPPRESS
                    kwargs["required"] = True
                elif f.default is MISSING:
                    kwargs["default"] = f.default_factory()
                    kwargs["help"] = HELP
                else:
                    kwargs["default"] = f.default
                    kwargs["help"] = HELP

                f_type = f.type

                if is_union(f_type):
                    f_args = set(get_args(f_type)) - {type(None)}
                    if len(f_args) == 1:
                        f_type = f_args.pop()

                if is_literal(f_type):
                    group.add_argument(f_flag, choices=get_args(f_type), **kwargs)
                else:
                    group.add_argument(f_flag, type=string_parser(f_type), **kwargs)

    def _finalize(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> ArgumentParser:
        clone = copy.deepcopy(self)
        visited = set()

        while True:
            with (
                patch("argparse.ArgumentParser.error"),
                patch("argparse._HelpAction.__call__"),
            ):
                namespace_clone, _ = argparse.ArgumentParser.parse_known_args(
                    clone, args=args, namespace=copy.copy(namespace)
                )

            finished = True
            for spec in clone.heracls_specs.values():
                for dest, table in spec.choices.items():
                    if dest not in visited:
                        visited.add(dest)
                        choice = getattr(namespace_clone, f"{dest}:choice")
                        value = table[choice]
                        if is_dataclass_type(value):
                            clone._register_dataclass(spec, value, dest.split("."))
                            finished = False

            if finished:
                break

        return clone

    def parse_known_args(
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        clone = self._finalize(args, namespace)

        namespace, unknown = argparse.ArgumentParser.parse_known_args(
            clone, args=args, namespace=namespace
        )

        dest_map = {}

        for dest, spec in clone.heracls_specs.items():
            for key in spec.keys:
                dest_map[key] = dest

        data_dicts = {dest: {} for dest in clone.heracls_specs}
        others = {}

        for key, value in vars(namespace).items():
            if key.endswith(":choice"):
                pass
            elif key in dest_map:
                data_dicts[dest_map[key]][key] = value
            else:
                others[key] = value

        for dest, spec in clone.heracls_specs.items():
            data = {}
            for key, value in data_dicts[dest].items():
                temp = data
                *keys, leaf = key.split(".")
                for k in keys:
                    temp = temp.setdefault(k, {})
                temp[leaf] = value

            others[dest] = from_dict(spec.data_cls, data)

        namespace = argparse.Namespace(**others)

        return namespace, unknown

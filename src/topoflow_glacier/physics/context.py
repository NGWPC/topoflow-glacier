"""A file to hold classes relating to the model's context, or "state". This idea is related to how the internal variables of a model can be imported/exported/saved."""

from collections.abc import Iterable, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict


class Var(BaseModel):
    """Context variable representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    unit: str
    value: np.ndarray


class Context:
    """A class to hold a collection of variables representing model conditions"""

    def __init__(self, vars: Iterable[Var]):
        self._name_mapping: dict[str, Var] = {var.name: var for var in vars}

    def unit(self, name: str) -> str:
        """Given a variable name, return its unit"""
        return self._name_mapping[name].unit

    def value(self, name: str) -> np.ndarray:
        """Given a variable name, return a value reference"""
        return self._name_mapping[name].value

    def value_at_indices(self, name: str, dest: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # This must copy into dest!!!
        assert dest.shape[0] >= indices.shape[0], "dest smaller than indices"
        src = self.value(name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = src[value_index]
        return dest

    def set_value(self, name: str, value: np.ndarray):
        self._name_mapping[name].value[:] = value

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray):
        assert src.shape[0] >= inds.shape[0], "inds larger than src"
        arr = self.value(name)
        for i in range(inds.shape[0]):
            arr[inds[i]] = src[i]

    def names(self) -> Iterable[str]:
        yield from self._name_mapping

    def vars(self) -> Iterable[Var]:
        yield from self._name_mapping.values()

    def __contains__(self, name: str) -> bool:
        """Return if the variable name is present in the collection."""
        return name in self._name_mapping

    def __iter__(self) -> Iterator[Var]:
        return iter(self.vars())

    def __len__(self) -> int:
        return len(self._name_mapping)


def build_context(vars: Iterable[tuple[str, str]]) -> Context:
    g = (Var(name=name, unit=unit, value=np.array([0.0], dtype=np.float64)) for (name, unit) in vars)
    return Context(vars=g)

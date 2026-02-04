import importlib.util
from collections.abc import Mapping, Sequence, Iterable
from typing import Any, TypeVar
import functools, operator

# lazy loader from https://stackoverflow.com/a/78312674/15673832
class LazyLoader:
    'thin shell class to wrap modules.  load real module on first access and pass thru'

    def __init__(self, modname):
        self._modname = modname
        self._mod = None

    def __getattr__(self, attr):
        'import module on first attribute access'

        try:
            return getattr(self._mod, attr)

        except Exception as e :
            if self._mod is None :
                # module is unset, load it
                self._mod = importlib.import_module (self._modname)
            else :
                # module is set, got different exception from getattr ().  reraise it
                raise e

        # retry getattr if module was just loaded for first time
        # call this outside exception handler in case it raises new exception
        return getattr (self._mod, attr)


T = TypeVar("T")
def reduce_dim(x:Iterable[Iterable[T]]) -> list[T]:
    """Reduces one level of nesting. Takes an iterable of iterables of X, and returns an iterable of X."""
    return functools.reduce(operator.iconcat, x, [])

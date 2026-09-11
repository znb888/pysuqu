"""Process-safe reuse of prepared native operators and scratch storage."""

from __future__ import annotations

import os
import threading
from types import BuiltinFunctionType


class NativeExecution:
    """Lazily prepare one native snapshot for an immutable numerical plan."""

    def __init__(self, *, defer_first=False):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._function = None
        self._handle = None
        self._defer_first = bool(defer_first)
        self._seen = False

    def _ensure_current_process(self):
        current_pid = os.getpid()
        if self._pid == current_pid:
            return
        self._lock = threading.Lock()
        self._function = None
        self._handle = None
        self._seen = False
        self._pid = current_pid

    def __getstate__(self):
        return {"defer_first": self._defer_first}

    def __setstate__(self, state):
        self.__init__(defer_first=state.get("defer_first", False))

    def invoke(self, function, args, options, initial, fallback):
        """Invoke a prepared native function, falling back when unsupported."""
        self._ensure_current_process()
        prepared_call = (
            getattr(function.__self__, "propagate_prepared", None)
            if isinstance(function, BuiltinFunctionType)
            else None
        )
        if not callable(prepared_call):
            return fallback(function, *args, **options)
        with self._lock:
            defer = self._defer_first and not self._seen
            self._seen = True
            if not defer and (self._function is not function or self._handle is None):
                handle = function(*args, **options, prepare_only=True)
                self._function = function
                self._handle = handle
            handle = self._handle
        if defer:
            result = fallback(function, *args, **options)
            result[2].setdefault("native_prepared", False)
            result[2].setdefault("native_plan_reused", False)
            return result
        return prepared_call(
            handle,
            initial,
            **{
                key: options[key]
                for key in (
                    "atol", "rtol", "max_steps", "store_trajectory",
                    "sparse_expm", "parallel",
                )
            },
        )


__all__ = ["NativeExecution"]

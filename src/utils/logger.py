"""Minimal, dependency-free console logging used by every pipeline stage."""

import time

_START = time.perf_counter()


def _stamp() -> str:
    return f"{time.perf_counter() - _START:7.1f}s"


def log_step(message: str) -> None:
    print(f"\n[{_stamp()}] ===== {message} =====")


def log_info(message: str) -> None:
    print(f"[{_stamp()}] [INFO] {message}")


def log_success(message: str) -> None:
    print(f"[{_stamp()}] [SUCCESS] {message}")


def log_warning(message: str) -> None:
    print(f"[{_stamp()}] [WARNING] {message}")

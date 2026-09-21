import threading

import pytest

from robotarm.deployment.monitored_work import run_monitored_work


def test_monitor_runs_while_compute_is_waiting_and_owns_caller_thread():
    release = threading.Event()
    owner = threading.get_ident()
    observed = []

    def work():
        assert threading.get_ident() != owner
        assert release.wait(5)
        return 42

    def monitor():
        assert threading.get_ident() == owner
        observed.append(True)
        release.set()

    assert run_monitored_work(work, monitor) == 42
    assert observed


def test_monitor_failure_returns_before_blocked_compute_for_hardware_shutdown():
    release = threading.Event()

    def work():
        release.wait(5)

    def abort():
        raise RuntimeError("lock deadline")

    try:
        with pytest.raises(RuntimeError, match="lock deadline"):
            run_monitored_work(work, abort)
        assert not release.is_set()
    finally:
        release.set()


def test_worker_failure_is_not_swallowed():
    def fail():
        raise ValueError("invalid scores")
    with pytest.raises(ValueError, match="invalid scores"):
        run_monitored_work(fail, lambda: None)

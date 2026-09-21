"""Run pure compute/file work while the caller retains exclusive hardware I/O."""
from concurrent.futures import ThreadPoolExecutor


def run_monitored_work(work, monitor):
    """Monitor may raise to enter hardware shutdown without waiting for compute.

    The worker must never access hardware or send commands. It may finish pure
    computation after cancellation; no result is consumed after monitor failure.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ipwm-compute")
    future = pool.submit(work)
    try:
        while not future.done():
            monitor()
        monitor()
        return future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

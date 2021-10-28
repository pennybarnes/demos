import math
import time


def wgs_to_epsg(lat, lon):
    """
    Get the epsg code from a (lat, lon) location
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band
    return epsg_code


def as_completed(jobs, interval_sec=5):
    """
    Iterator over Jobs that yields each Job when it completes.

    Parameters
    ----------
    jobs: Sequence[wf.Job]
        Jobs to wait for
    interval_sec: int, optional, default 5
        Wait at least this many seconds between polling for job updates.

    Yields
    ------
    job: wf.Job
        A completed job (either succeeded or failed).
    """
    jobs = list(jobs)
    while len(jobs) > 0:
        loop_start = time.perf_counter()

        i = 0
        while i < len(jobs):
            job = jobs[i]
            if not job.done:  # in case it's already loaded
                try:
                    job.refresh()
                except Exception:
                    continue  # be resilient to transient errors for now

            if job.done:
                yield job
                del jobs[i]  # "advances" i
            else:
                i += 1

        loop_duration = time.perf_counter() - loop_start
        if len(jobs) > 0 and loop_duration < interval_sec:
            time.sleep(interval_sec - loop_duration)

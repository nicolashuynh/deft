def run_baseline_experiment(*args, **kwargs):
    """Run baseline experiment."""
    from src.baselines.runner import run_baseline_experiment as _impl

    return _impl(*args, **kwargs)


def run_dl_baselines_experiment(*args, **kwargs):
    """Run dl baselines experiment."""
    from src.baselines.dl_runner import run_dl_baselines_experiment as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "run_baseline_experiment",
    "run_dl_baselines_experiment",
]

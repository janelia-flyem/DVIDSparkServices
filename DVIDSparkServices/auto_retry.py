import functools
def auto_retry(total_tries=1, pause_between_tries=10.0, logger=None):
    """
    Returns a decorator.
    If the decorated function fails for any reason,
    pause for a bit and then retry until it has been called total_tries times.
    """
    assert total_tries >= 1
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            remaining_tries = total_tries
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    remaining_tries -= 1
                    if total_tries == 0:
                        raise
                    if logger:
                        logger.warn("Call to '{}' failed with error: {}.".format(func.func_name, repr(ex)))
                        logger.warn("Retrying {} more times".format( remaining_tries ))
                    import time
                    time.sleep(pause_between_tries)
        wrapper.__wrapped__ = func # Emulate python 3 behavior of @functools.wraps
        return wrapper
    return decorator

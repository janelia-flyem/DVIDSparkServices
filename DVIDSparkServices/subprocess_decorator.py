from __future__ import print_function, absolute_import
import sys
import time
import subprocess
import tempfile
import shutil
import pickle as pickle
import logging
import threading

def execute_in_subprocess( stdout_callback=None, timeout=0 ):
    """
    Returns a decorator-like function, but you can't actually use 
    decorator syntax (@) with it, for technical reasons.
    
    When called, the "decorated" function is executed in a subprocess.
    The arguments and return value are passed via pickling.
    
    Note: The function you intend to execute must be declared at a module
    top-level scope (requirement of pickle).
    
    TODO: For now, arguments are passed via a temporary file instead of in
          memory over a socket or a pipe.
        
    stdout_callback: The subprocess output is captured and passed to
                     the given callback, line-by-line.
                     If no callback is provided, the output is sent
                     to the python logging module by default.
                     (In the subprocess, stdout and stderr are merged into the same stream.)

    Example:
    
        def foo(a,b,c):
            print a,b,c
        
        foo_in_subprocess = execute_in_subprocess(logger.info)(foo)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            _stdout_callback = stdout_callback
            if _stdout_callback is None:
                logger = logging.getLogger(__name__ + '.' + func.__name__)
                logger.setLevel(logging.INFO)
                _stdout_callback = logger.info

            tmpdir = tempfile.mkdtemp()
            args_filepath = tmpdir + '/{}-input-args.pkl'.format(func.__name__)
            result_filepath = tmpdir + '/{}-result.pkl'.format(func.__name__)
            try:
                # Write func/args to file
                with Timer() as timer:
                    with open(args_filepath, 'w') as args_f:
                        pickle.dump((func, args, kwargs), args_f, protocol=2)
                _stdout_callback("Serializing args took: {:.03f}\n".format(timer.seconds))

                # Start process                        
                with Timer() as timer:
                    p = subprocess.Popen([sys.executable, __file__, args_filepath, result_filepath],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                _stdout_callback("Process startup took: {:.03f}\n".format(timer.seconds))

                # Start timeout thread
                killed_early = [False]
                if timeout:
                    def watchdog():
                        start = time.time()
                        while True:
                            if p.poll() is not None:
                                return
                            if (time.time() - start) > timeout:
                                break
                            time.sleep(1.0)

                        # Uh-oh: timed out
                        killed_early[0] = True
                        p.kill()

                    threading.Thread(target=watchdog).start()

                # Wait for process to finish        
                while True:
                    line = p.stdout.readline()
                    if line == '' and p.poll() is not None:
                        break
                    _stdout_callback(line)

                # Check for timeout
                if killed_early[0]:
                    raise RuntimeError("Killed '{}' subprocess after timeout of {} seconds.".format(func.__name__, timeout))

                # Read result
                with Timer() as timer:
                    with open(result_filepath, 'r') as result_f:
                        result = pickle.load(result_f)
                _stdout_callback("Deserializing result took: {:.03f}\n".format(timer.seconds))

                # Check for other errors                
                if isinstance(result, Exception):
                    raise RuntimeError("Failed to execute '{}' in a subprocess. Examine subprocess output for Traceback.".format(func.__name__))

                return result
            finally:
                # Clean up
                shutil.rmtree(tmpdir)
        return wrapper
    return decorator


class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.seconds = time.time() - self.start

def subprocess_main():
    args_filepath = sys.argv[1]
    result_filepath = sys.argv[2]

    with Timer() as timer:
        with open(args_filepath, 'r') as args_f:
            func, args, kwargs = pickle.load(args_f)
    print("Deserializing args took: {:.03f}".format(timer.seconds))

    try:    
        with Timer() as timer:
            result = func(*args, **kwargs)
        print("Function execution took: {:.03f}".format(timer.seconds))

    except Exception as ex:
        result = ex
        raise

    finally:
        with Timer() as timer:
            with open(result_filepath, 'w') as result_f:
                pickle.dump(result, result_f, protocol=2)
        print("Serializing result took: {:.03f}".format(timer.seconds))

def test_helper(a,b,c):
    """
    This function is just here for the unit test to call.
    (It can't be defined in the test module due to edge
    cases involving modules named '__main__'.)
    """
    print(a)
    print(b)
    print(c)
    time.sleep(c)
    return a + b + c

if __name__ == "__main__":
    sys.exit( subprocess_main() )

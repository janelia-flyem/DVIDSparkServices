import os
import sys
import time
import signal
import pickle
import inspect
import logging
import threading
from functools import partial
from contextlib import contextmanager
from multiprocessing import Pool, TimeoutError, Pipe

import ctypes
from ctypes.util import find_library

try:
    libc = ctypes.cdll.msvcrt # Windows
except OSError:
    libc = ctypes.cdll.LoadLibrary(find_library('c'))


def execute_in_subprocess(timeout=None, stream_logger=None):
    """
    Excecutes the "decorated" function in a subprocess,
    waits for the result, and returns it.

    Note: Returns a pseudo-decorator.
          For technical reasons, you can't actually use Python @-syntax.
    
    Example:
        
        def myfunc(a,b,c):
            print("yada yada")
            logging.getLogger().info("YADA YADA")
            return 'yada yada'
        
        myfunc_in_subprocess = execute_in_subprocess(10.0, logger)(myfunc)
        result = myfunc_in_subprocess('a', 'b', 'c')
    
    Features:
    
    - A timeout can be specified, after which the subprocess is killed and a TimeoutError is raised.

    - All Python logging messages in the subprocess are suppressed in the
      subprocess and logged in the parent instead. 

    - Optionally, all stdout and stderr output in the subprocess can be captured
      in the parent as logging messages (instead of printed to the parent stdout/stderr).

    - Exceptions in the subprocess are forwarded to the parent process
    
    - On Unix, uses fork to avoid interpreter startup overhead.
      (Not tested on Windows, which uses 'spawn'.)
    
    Args:
    
        timeout: (float)
                 Seconds after which the process should be
                 killed and a multiprocessing.TimeoutError is raised.
        
        stream_logger: (logging.Logger)
                        If provided, all stderr/stdout in the subprocess will be
                        suppressed and instead logged (line-by-line) to the given logger object.
                        If not provided, the subprocess inherits the parent's stderr and stdout as usual.
    
    """
    assert timeout >= 1.0, \
        f"Timeout is too short: ({timeout}).\n"\
        "The execute_in_subprocess() decorator does not behave well for very short timeouts.\n" \
        "It is intended to be used as a safeguard for potentially long-running jobs."
    def decorator(func):
        try:
            pickle.dumps(func)
        except Exception:
            raise RuntimeError(f"Can't decorate this function ({func.__name__}) with execute_in_subprocess():\n"
                               "It isn't pickleable. Try declaring the function at module scope.")
        
        def wrapper(*args, **kwargs):
            func_with_args = partial( func, *args, **kwargs )
            
            # Open pipes for the child process to inherit.
            log_records_from_child, log_records_to_parent = Pipe(False)
            stdout_from_child, stdout_to_parent = os.pipe()
            stderr_from_child, stderr_to_parent = os.pipe()
            
            # Fork (via a Pool with a single process)
            pool = Pool(1)

            # After fork, must close unused ends of the pipes,
            # on both child and parent.
            os.close( stdout_to_parent )
            os.close( stderr_to_parent )
            pool.apply( os.close, (stdout_from_child,) )
            pool.apply( os.close, (stderr_from_child,) )

            # This thread captures any Python log messages in the subprocess.        
            record_logging_thread = _ChildLogEchoThread(log_records_from_child)
            record_logging_thread.start()

            if stream_logger:
                # These threads capture any stdout/stderr output from the subprocess.
                stdout_logging_thread = _ChildStreamLoggingThread(stream_logger, logging.INFO,  stdout_from_child, inspect.stack()[1])
                stderr_logging_thread = _ChildStreamLoggingThread(stream_logger, logging.ERROR, stderr_from_child, inspect.stack()[1])

                stdout_logging_thread.start()
                stderr_logging_thread.start()
            else:
                # No stream_logger given.
                # Child will just inherit parent stdout/stderr as usual
                stdout_to_parent = sys.stdout.fileno()
                stderr_to_parent = sys.stderr.fileno()
            
            child_pid = pool.apply(os.getpid)
            future = pool.apply_async(_subproc_inner_wrapper, (func_with_args, stdout_to_parent, stderr_to_parent, log_records_to_parent))

            try:
                return future.get(timeout)
            except TimeoutError:
                try:
                    # We timed out.
                    # Kill the child process and make sure it's REALLY dead
                    # Don't use pool._pool[0].terminate() or .kill() because that can hang...
                    os.kill(child_pid, signal.SIGTERM)
                    time.sleep(1.0)
                    os.kill(child_pid, signal.SIGKILL) 
                    os.waitpid(child_pid, 0)
                except Exception:
                    pass
                raise
            else:
                # These are already invalid if the process was killed due to timeout.
                os.close(stdout_from_child)
                os.close(stderr_from_child)
                
            finally:
                pool.terminate()
                pool.close()
                pool.join()
                del pool

                record_logging_thread.stop_and_join()
                if stream_logger:
                    stdout_logging_thread.join()
                    stderr_logging_thread.join()

        return wrapper
    return decorator


def _subproc_inner_wrapper( func, stdout_fd, stderr_fd, connection_to_parent ):
    """
    Helper function.
    Runs within the child process, and calls the user's function within a
    special context so that log messages and stdout/stderr are redirected.
    """
    # Disable all root logging handlers and install a new
    # one that just forwards log messages to the parent.
    handler = _ForkedProcessLogHandler(connection_to_parent)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    try:    
        # redirect stdout and stderr to pipes for the parent to pull from.
        with stdout_redirected( stdout_fd, sys.stdout ), \
             stdout_redirected( stderr_fd, sys.stderr ):

            try:
                return func()
            finally:
                # I don't understand why these flush calls are necessary, 
                # but without them the last line can get dropped if it doesn't end with '\n' 
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.
    
    Redirects a file object or file descriptor to a new file descriptor.
    
    Motivation: In pure-Python, you can redirect all print() statements like this:
        
        sys.stdout = open('myfile.txt')
        
        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.

    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.
    
    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    if fileno(to) == stdout_fd:
        # Nothing to do; early return
        yield stdout
        return

    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        libc.fflush(None) # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


class _ForkedProcessLogHandler(logging.Handler):
    """
    A logging.Handler to be used within forked processes.
    Log records are handled by sending them to the parent
    process via a multiprocessing.Connection.
    """
    def __init__(self, connection_to_parent, level=logging.NOTSET):
        """
        connection_to_parent: multiprocessing.Connection.  The parent process should 
        """
        super().__init__(level)
        self._connection = connection_to_parent
        
    def emit(self, record):
        self._connection.send(record)


class _ChildLogEchoThread(threading.Thread):
    """
    Thread that pulls logging records from a subprocess (via a multiprocessing.Connection object),
    and emits those log messages in this parent process as if they were generated here.
    """
    def __init__(self, connection_from_child):
        """
        connection_from_child:
            A readable multiprocess.Connection object which yields logging.LogRecord objects.
        """
        super().__init__()
        self.daemon = True
        self.__stop = False
        self.__connection = connection_from_child
    
    def run(self):
        while not self.__stop:
            if self.__connection.poll(0.1):
                record = self.__connection.recv()
                logging.getLogger(record.name).handle(record)

    def stop_and_join(self):
        self.__stop = True
        self.join()


class _ChildStreamLoggingThread(threading.Thread):
    """
    Thread that listens to a file descriptor and logs each line as it comes in.
    Each line is logged as a separate logging message. 
    """
    def __init__(self, logger, level, pipe_from_child_fd, frame_info=None):
        super().__init__()
        self.daemon = True
        self.logger = logger
        self.level = level
        self.__fd = pipe_from_child_fd
        self.frame_info = frame_info

    def run(self):
        with os.fdopen(self.__fd, 'r') as f:
            for stream_line in f:
                if self.frame_info is None:
                    self.logger.log(self.level, stream_line.rstrip())
                else:
                    record = self.logger.makeRecord( self.logger.name,
                                                     self.level,
                                                     self.frame_info.filename,
                                                     self.frame_info.lineno,
                                                     stream_line.rstrip(),
                                                     (),
                                                     None,
                                                     self.frame_info.function )
                    self.logger.handle(record)


def _test_helper(a,b,c):
    """
    This function is just here for the unit test to call.
    (It can't be defined in the test module due to edge
    cases involving modules named '__main__'.)
    """
    print(a)
    print(b, file=sys.stderr)
    print(c, end='') # Verify that everything works even if the last line doesn't end with '\n'
    time.sleep(c)
    return a + b + c

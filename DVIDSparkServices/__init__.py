from __future__ import print_function, absolute_import
import os
if "DVIDSPARK_WORKFLOW_TMPDIR" in os.environ and os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]:
    # Override the tempfile location for all python functions
    import tempfile
    tempfile.tempdir = os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]

import sys
from os import makedirs
import threading
import traceback
import logging
import signal
from io import StringIO
import socket
from pathlib import Path
import glob
import faulthandler
import contextlib
from subprocess import Popen, PIPE
import atexit

# Segfaults will trigger a traceback dump
FAULTHANDLER_TEE, FAULTHANDLER_OUTPUT_PATH = None, None
def setup_faulthandler():
    """
    Enable the faulthandler module so that it dumps tracebacks.
    We use the unix 'tee' command send it to both stderr AND a file on disk.
    """
    global FAULTHANDLER_TEE
    global FAULTHANDLER_OUTPUT_PATH

    output_dir = os.environ.get("DVIDSPARKSERVICES_FAULTHANDLER_OUTPUT_DIR", "")
    if not output_dir:
        import getpass
        output_dir = f"/tmp/{getpass.getuser()}"

    makedirs(output_dir, exist_ok=True)

    # All worker processes will send faulthandler output to the same place,
    # but in practice the output shouldn't be too jumbled -- they are unlikely
    # to segfault at exactly the same moment.
    FAULTHANDLER_OUTPUT_PATH = output_dir + '/_FAULTHANDLER_OUTPUT_.log'

    FAULTHANDLER_TEE = Popen(f'tee -a {FAULTHANDLER_OUTPUT_PATH}', shell=True, stdin=PIPE)
    faulthandler.enable(FAULTHANDLER_TEE.stdin)

# We enable faulthandler here, as soon as DVIDSparkServices is imported,
# so that all spark tasks use it automatically.
setup_faulthandler()

@contextlib.contextmanager
def pause_faulthandler():
    faulthandler.disable()
    yield
    faulthandler.enable(FAULTHANDLER_TEE.stdin)

# Cleanup function is intended for Workflow
def cleanup_faulthandler(atexit=False):
    """
    Disable the faulthandler module, and delete the on-disk log file if it's empty.
    Only call this from the driver, just before it exits.
    
    atexit: If True, stop the 'tee' process, but don't overwrite the output file.
    """
    # Disable before closing the file handle.
    faulthandler.disable()
    
    # May as well re-enable it for stderr output, at least.
    faulthandler.enable()

    # Close the file handle
    if FAULTHANDLER_TEE:
        FAULTHANDLER_TEE.stdin.close()
        FAULTHANDLER_TEE.wait()
    
    if atexit:
        return

    # Remove the file if it's empty.
    if os.path.exists(FAULTHANDLER_OUTPUT_PATH):
        if os.stat(FAULTHANDLER_OUTPUT_PATH).st_size == 0:
            os.unlink(FAULTHANDLER_OUTPUT_PATH)
        else:
            sys.stderr.write("*****************************************\n")
            sys.stderr.write("SEGFAULT DETECTED IN ONE OR MORE WORKERS:\n")
            with open(FAULTHANDLER_OUTPUT_PATH, 'r') as f:
                sys.stderr.write(f.read())
            sys.stderr.write("*****************************************\n")

# Make sure that the 'tee' process is not still running while we shut down.
# Under normal circumstances, the driver calls cleanup_faulthandler() in Workflow.run(),
# but if no workflow was run (e.g. during testing), we still need to call it.
# Otherwise, we get bitten by this bug:
# - https://bugs.python.org/issue26153
atexit.register(cleanup_faulthandler, True)

# Ensure SystemExit is raised if terminated via SIGTERM (e.g. by bkill).
signal.signal(signal.SIGTERM, lambda signum, stack_frame: sys.exit(0))

# Ensure SystemExit is raised if terminated via SIGUSR2.
# (The LSF cluster scheduler uses SIGUSR2 if the job's -W time limit has been exceeded.)
signal.signal(signal.SIGUSR2, lambda signum, stack_frame: sys.exit(0))
    
formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(module)s %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('requests').setLevel('DEBUG')
logging.getLogger('DVIDSparkServices.dvid.metadata').setLevel(logging.DEBUG)


def initialize_excepthook():
    """
    This excepthook simply logs all unhandled exception tracebacks with Logger.error()
    """
    sys.excepthook = _log_exception
    _install_thread_excepthook()

def _log_exception(*exc_info):
    thread_name = threading.current_thread().name
    logging.getLogger().error( "Unhandled exception in thread: '{}'".format(thread_name) )
    sio = StringIO()
    traceback.print_exception( exc_info[0], exc_info[1], exc_info[2], file=sio )
    logging.getLogger().error( sio.getvalue() )

def _install_thread_excepthook():
    # This function was copied from: http://bugs.python.org/issue1230540
    # It is necessary because sys.excepthook doesn't work for unhandled exceptions in other threads.
    """
    Workaround for sys.excepthook thread bug
    (https://sourceforge.net/tracker/?func=detail&atid=105470&aid=1230540&group_id=5470).
    Call once from __main__ before creating any threads.
    If using psyco, call psycho.cannotcompile(threading.Thread.run)
    since this replaces a new-style class method.
    """
    run_old = threading.Thread.run
    def run(*args, **kwargs):
        try:
            run_old(*args, **kwargs)
        except SystemExit as ex:
            if ex.code != 0:
                sys.excepthook(*sys.exc_info())
                raise
        except:
            sys.excepthook(*sys.exc_info())
            raise
    threading.Thread.run = run


initialize_excepthook()


# Activate compressed numpy pickling in all workflows
from .sparkdvid.CompressedNumpyArray import activate_compressed_numpy_pickling
activate_compressed_numpy_pickling()

def connect_debugger():
    import sys
    import os
    
    # Possible paths to the pydev debugger module on your hard drive.
    # Developers: Add your dev machine's pydev directory to this list.
    pydev_src_paths = [ "/Applications/eclipse/plugins/org.python.pydev_4.5.5.201603221110/pysrc/",
                        "/usr/local/eclipse/plugins/org.python.pydev_4.2.0.201507041133/pysrc/",
                        '/Users/bergs/.p2/pool/plugins/org.python.pydev_5.5.0.201701191708/pysrc/' ]

    pydev_src_paths = list(filter(os.path.exists, pydev_src_paths))
    
    if not pydev_src_paths:
        raise RuntimeError("Error: Couldn't find the path to the pydev module.  You can't use PYDEV_DEBUGGER_ENABLED.")
    
    if len(pydev_src_paths) > 1:
        raise RuntimeError("Error: I found more than one pydev module.  I don't know which one to use.")
    
    sys.path.append(pydev_src_paths[0])
    import pydevd
    print("Waiting for PyDev debugger...")
    pydevd.settrace(stdoutToServer=True, stderrToServer=True, suspend=False)

if int(os.getenv('PYDEV_DEBUGGER_ENABLED', 0)):
    connect_debugger()

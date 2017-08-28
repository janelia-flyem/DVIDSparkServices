from __future__ import print_function, absolute_import
import os
if "DVIDSPARK_WORKFLOW_TMPDIR" in os.environ and os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]:
    # Override the tempfile location for all python functions
    import tempfile
    tempfile.tempdir = os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]

import sys
import threading
import traceback
import logging

from io import StringIO
    
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
        #except (KeyboardInterrupt, SystemExit):
        #    raise
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

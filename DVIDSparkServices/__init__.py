import os
if "DVIDSPARK_WORKFLOW_TMPDIR" in os.environ and os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]:
    # Override the tempfile location for all python functions
    import tempfile
    tempfile.tempdir = os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]

# Activate compressed numpy pickling in all workflows
from .sparkdvid.CompressedNumpyArray import activate_compressed_numpy_pickling
activate_compressed_numpy_pickling()

def connect_debugger():
    import sys
    import os
    
    # Possible paths to the pydev debugger module on your hard drive.
    # Developers: Add your dev machine's pydev directory to this list.
    pydev_src_paths = [ "/Applications/eclipse/plugins/org.python.pydev_4.5.5.201603221110/pysrc/",
                        "/usr/local/eclipse/plugins/org.python.pydev_4.2.0.201507041133/pysrc/" ]

    pydev_src_paths = filter(os.path.exists, pydev_src_paths)
    
    if not pydev_src_paths:
        raise RuntimeError("Error: Couldn't find the path to the pydev module.  You can't use PYDEV_DEBUGGER_ENABLED.")
    
    if len(pydev_src_paths) > 1:
        raise RuntimeError("Error: I found more than one pydev module.  I don't know which one to use.")
    
    sys.path.append(pydev_src_paths[0])
    import pydevd
    print "Waiting for PyDev debugger..."
    pydevd.settrace(stdoutToServer=True, stderrToServer=True, suspend=False)

PYDEV_DEBUGGER_ENABLED = False
if PYDEV_DEBUGGER_ENABLED:
    connect_debugger()

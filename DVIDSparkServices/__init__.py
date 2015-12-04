def connect_debugger():
    import sys
    import os
    pydev_src_path = "/Applications/eclipse/plugins/org.python.pydev_4.1.0.201505270003/pysrc"
    if not os.path.exists(pydev_src_path):
        raise RuntimeError("Error: Couldn't find the path to the pydev module.  You can't use PYDEV_DEBUGGER_ENABLED.")
    sys.path.append(pydev_src_path)
    import pydevd
    print "Waiting for PyDev debugger..."
    pydevd.settrace(stdoutToServer=True, stderrToServer=True, suspend=False)

PYDEV_DEBUGGER_ENABLED = False
if PYDEV_DEBUGGER_ENABLED:
    connect_debugger()

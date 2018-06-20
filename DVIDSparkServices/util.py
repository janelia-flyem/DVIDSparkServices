from __future__ import print_function, absolute_import
from __future__ import division
import os
import sys
from signal import SIGINT, SIGTERM, SIGKILL
import copy
import time
import contextlib
import inspect
import socket
import logging
from itertools import starmap, product
from datetime import timedelta
import json
import getpass
import threading

import requests
import psutil
import numpy as np
from skimage.util import view_as_blocks

# Don't import pandas here; import it locally as needed
#import pandas as pd

from numba import jit
from psutil import AccessDenied

logger = logging.getLogger(__name__)

def cpus_per_worker():
    return 16

DEFAULT_DVID_SESSIONS = {}
DEFAULT_APPNAME = "DVIDSparkServices" # Workflows should overwrite this in their constructors.

def default_dvid_session(appname=None):
    """
    Return a default requests.Session() object that automatically appends the
    'u' and 'app' query string parameters to every request.
    """
    if appname is None:
        appname = DEFAULT_APPNAME
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    try:
        s = DEFAULT_DVID_SESSIONS[(appname, thread_id)]
    except KeyError:
        s = requests.Session()
        s.params = { 'u': getpass.getuser(),
                     'app': appname }
        DEFAULT_DVID_SESSIONS[(appname, thread_id)] = s

    return s

def num_worker_nodes():
    if "NUM_SPARK_WORKERS" not in os.environ:
        # See sparklaunch_janelia_lsf
        raise RuntimeError("Error: Your driver launch script must define NUM_SPARK_WORKERS in the environment.")

    num_nodes = int(os.environ["NUM_SPARK_WORKERS"])
    num_workers = max(1, num_nodes) # Don't count the driver, unless it's the only thing
    return num_workers
    

def persist_and_execute(rdd, description, logger=None, storage=None):
    if storage is None:
        from pyspark import StorageLevel
        storage = StorageLevel.MEMORY_ONLY

    if logger:
        logger.info(f"{description}...")

    with Timer() as timer:
        rdd.persist(storage)
        count = rdd.count()

    if logger:
        logger.info(f"{description} (N={count}) took {timer.timedelta}")
    
    return count

def unpersist(rdd):
    # PySpark freaks out if you try to unpersist an RDD
    # that wasn't persisted in the first place
    if rdd.is_cached:
        rdd.unpersist()

def get_localhost_ip_address():
    """
    Return this machine's own IP address, as seen from the network
    (e.g. 192.168.1.152, not 127.0.0.1)
    """
    try:
        # Determine our own machine's IP address
        # This method is a little hacky because it requires
        # making a connection to some arbitrary external site,
        # but it seems to be more reliable than the method below. 
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("google.com",80))
        ip_addr = s.getsockname()[0]
        s.close()
        
    except socket.gaierror:
        # Warning: This method is simpler, but unreliable on some networks.
        #          For example, on a home Verizon FiOS network it will error out in the best case,
        #          or return the wrong IP in the worst case (if you haven't disabled their DNS
        #          hijacking on your router)
        ip_addr = socket.gethostbyname(socket.gethostname())
    
    return ip_addr
    

class NumpyConvertingEncoder(json.JSONEncoder):
    """
    Encoder that converts numpy arrays and scalars
    into their pure-python counterparts.
    
    (No attempt is made to preserve bit-width information.)
    
    Usage:
    
        >>> d = {"a": np.arange(3, dtype=np.uint32)}
        >>> json.dumps(d, cls=NumpyConvertingEncoder)
        '{"a": [0, 1, 2]}'
    """
    def default(self, o):
        if isinstance(o, (np.ndarray, np.number)):
            return o.tolist()
        return super().default(o)

@contextlib.contextmanager
def Timer(msg=None, logger=None):
    if msg:
        logger = logger or logging.getLogger(__name__)
        logger.info(msg + '...')
    result = _TimerResult()
    start = time.time()
    yield result
    result.seconds = time.time() - start
    result.timedelta = timedelta(seconds=result.seconds)
    if msg:
        logger.info(msg + f' took {result.timedelta}')

class _TimerResult(object):
    seconds = -1.0

def line_number():
    """
    Return the currently executing line number in the caller's function.
    """
    return inspect.currentframe().f_back.f_lineno

class MemoryWatcher(object):
    def __init__(self, threshold_mb=1.0):
        self.hostname = socket.gethostname().split('.')[0]
        self.current_process = psutil.Process()
        self.initial_memory_usage = -1
        self.threshold_mb = threshold_mb
        self.ignore_threshold = False
    
    def __enter__(self):
        self.initial_memory_usage = self.current_process.memory_info().rss
        return self
    
    def __exit__(self, *args):
        pass

    def memory_increase(self):
        return self.current_process.memory_info().rss - self.initial_memory_usage
    
    def memory_increase_mb(self):
        return self.memory_increase() / 1024.0 / 1024.0

    def log_increase(self, logger, level=logging.DEBUG, note=""):
        if logger.isEnabledFor(level):
            caller_line = inspect.currentframe().f_back.f_lineno
            caller_file = os.path.basename( inspect.currentframe().f_back.f_code.co_filename )
            increase_mb = self.memory_increase_mb()
            
            if increase_mb > self.threshold_mb or self.ignore_threshold:
                # As soon as any message exceeds the threshold, show all messages from then on.
                self.ignore_threshold = True
                logger.log(level, "Memory increase: {:.1f} MB [{}] [{}:{}] ({})"
                                  .format(increase_mb, self.hostname, caller_file, caller_line, note) )


def unicode_to_str(json_data):
    if sys.version_info.major > 2:
        # In Python 3, unicode and str are the same
        return json_data

    # Python 2
    if isinstance(json_data, unicode):
        return str(json_data)
    elif isinstance(json_data, list):
        return map(unicode_to_str, json_data)
    elif isinstance(json_data, dict):
        json_data = copy.deepcopy(json_data)
        for k,v in json_data.items():
            json_data[k] = unicode_to_str(v)
        return json_data
    else:
        return json_data

def bb_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )


def extract_subvol(array, box):
    return array[box_to_slicing(*box)]

def overwrite_subvol(array, box, subarray):
    try:
        array[box_to_slicing(*box)] = subarray
    except:
        assert (subarray.shape == box[1] - box[0]).all(), \
            f"subarray is the wrong shape {subarray.shape} for the given box {box}"
        raise

def bb_as_tuple(box):
    if isinstance(box, np.ndarray):
        box = box.tolist()
    return (tuple(box[0]), tuple(box[1]))

# Aliases
box_to_slicing = bb_to_slicing
box_as_tuple = bb_as_tuple

def box_intersection(box_A, box_B):
    """
    Compute the intersection of the two given boxes.
    If the two boxes do not intersect at all, then the returned box will have non-positive shape:

    >>> intersection = box_intersection(box_A, box_B)
    >>> assert (intersection[1] - intersection[0] > 0).all(), "Boxes do not intersect."
    """
    intersection = np.empty_like(box_A)
    intersection[0] = np.maximum( box_A[0], box_B[0] )
    intersection[1] = np.minimum( box_A[1], box_B[1] )
    return intersection


def ndrange(start, stop=None, step=None):
    """
    Generator.

    Like np.ndindex, but accepts start/stop/step instead of
    assuming that start is always (0,0,0) and step is (1,1,1).
    
    Example:
    
    >>> for index in ndrange((1,2,3), (10,20,30), step=(5,10,15)):
    ...     print(index)
    (1, 2, 3)
    (1, 2, 18)
    (1, 12, 3)
    (1, 12, 18)
    (6, 2, 3)
    (6, 2, 18)
    (6, 12, 3)
    (6, 12, 18)
    """
    if stop is None:
        stop = start
        start = (0,)*len(stop)

    if step is None:
        step = (1,)*len(stop)

    assert len(start) == len(stop) == len(step), \
        f"tuple lengths don't match: ndrange({start}, {stop}, {step})"

    for index in product(*starmap(range, zip(start, stop, step))):
        yield index

def boxlist_to_json( bounds_list, indent=0 ):
    # The 'json' module doesn't have nice pretty-printing options for our purposes,
    # so we'll do this ourselves.
    from io import StringIO

    buf = StringIO()
    buf.write('    [\n')
    
    bounds_list, last_item = bounds_list[:-1], bounds_list[-1:]
    
    for bounds_zyx in bounds_list:
        start_str = '[{}, {}, {}]'.format(*bounds_zyx[0])
        stop_str  = '[{}, {}, {}]'.format(*bounds_zyx[1])
        buf.write(' '*indent + '[ ' + start_str + ', ' + stop_str + ' ],\n')

    # Write last entry
    if last_item:
        last_item = last_item[0]
        start_str = '[{}, {}, {}]'.format(*last_item[0])
        stop_str  = '[{}, {}, {}]'.format(*last_item[1])
        buf.write(' '*indent + '[ ' + start_str + ', ' + stop_str + ' ]')

    buf.write('\n')
    buf.write(' '*indent + ']')

    return str(buf.getvalue())

def replace_default_entries(array, default_array, marker=-1):
    """
    Overwrite all entries in array that match the given
    marker with the corresponding entry in default_array.
    """
    new_array = np.array(array)
    default_array = np.asarray(default_array)
    assert new_array.shape == default_array.shape
    new_array[:] = np.where(new_array == marker, default_array, new_array)
    
    if isinstance(array, np.ndarray):
        array[:] = new_array
    elif isinstance(array, list):
        # Slicewise assignment is broken for Ruamel sequences,
        # which are often passed to this function.
        # array[:] = new_array.list() # <-- broken
        # https://bitbucket.org/ruamel/yaml/issues/176/commentedseq-does-not-support-slice
        #
        # Use one-by-one item assignment instead:
        for i,val in enumerate(new_array.tolist()):
            array[i] = val
    else:
        raise RuntimeError("This function supports arrays and lists, nothing else.")

def mkdir_p(path):
    """
    Like the bash command: mkdir -p
    
    ...why the heck isn't this built-in to the Python std library?
    """
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def kill_if_running(pid, escalation_delay_seconds=10.0):
    """
    Kill the given process if it is still running.
    The process will be sent SIGINT, then SIGTERM if
    necessary (after escalation_delay seconds)
    and finally SIGKILL if it still hasn't died.
    
    This is similar to the behavior of the LSF 'bkill' command.
    
    Returns: True if the process was terminated 'nicely', or
             False it if it had to be killed with SIGKILL.
    """
    try:
        proc_cmd = ' '.join( psutil.Process(pid).cmdline() )
    except (psutil.NoSuchProcess, PermissionError, AccessDenied):
        return True

    _try_kill(pid, SIGINT)
    if not _is_still_running_after_delay(pid, escalation_delay_seconds):
        logger.info("Successfully interrupted process {}".format(pid))
        logger.info("Interrupted process was: " + proc_cmd)
        return True

    _try_kill(pid, SIGTERM)
    if not _is_still_running_after_delay(pid, escalation_delay_seconds):
        logger.info("Successfully terminated process {}".format(pid))
        logger.info("Terminated process was: " + proc_cmd)
        return True

    logger.warn("Process {} did not respond to SIGINT or SIGTERM.  Killing!".format(pid))
    
    # No more Mr. Nice Guy
    _try_kill(pid, SIGKILL, kill_children=True)
    logger.warn("Killed process was: " + proc_cmd)
    return False


def _try_kill(pid, sig, kill_children=False):
    """
    Attempt to terminate the process with the given ID via os.kill() with the given signal.
    If kill_children is True, then all child processes (and their children) 
    will be sent the signal as well, in unspecified order.
    """
    proc = psutil.Process(pid)
    procs_to_kill = [proc]
    
    if kill_children:
        for child in proc.children(recursive=True):
            procs_to_kill.append(child)
    
    for proc in procs_to_kill:
        try:
            os.kill(proc.pid, sig)
        except OSError as ex:
            if ex.errno != 3: # "No such process"
                raise

def _is_still_running_after_delay(pid, secs):
    still_running = is_process_running(pid)
    while still_running and secs > 0.0:
        time.sleep(2.0)
        secs -= 2.0
        still_running = is_process_running(pid)
    return still_running
    
def is_process_running(pid, reap_zombie=True):
    """
    Return True if a process with the given PID
    is currently running, False otherwise.
    
    reap_zombie:
        If it's in zombie state, reap it and return False.
    """
    # Sending signal 0 to a pid will raise an OSError 
    # exception if the pid is not running, and do nothing otherwise.        
    # https://stackoverflow.com/a/568285/162094
    try:
        os.kill(pid, 0) # Signal 0
    except OSError:
        return False

    if not reap_zombie:
        return True

    proc = psutil.Process(pid)
    if proc.status() != psutil.STATUS_ZOMBIE:
        return True

    # Process is a zombie. Attempt to reap.
    try:
        os.waitpid(proc.pid, 0)
    except ChildProcessError:
        pass

    return False

class RoiMap(object):
    """
    Little utility class to help with ROI manipulations
    """
    def __init__(self, roi_blocks):
        # Make a map of the entire ROI
        # Since roi blocks are 32^2, this won't be too huge.
        # For example, a ROI that's 10k*10k*100k pixels, this will be ~300 MB
        # For a 100k^3 ROI, this will be 30 GB (still small enough to fit in RAM on the driver)
        block_mask, (blocks_start, blocks_stop) = coordlist_to_boolmap(roi_blocks)
        blocks_shape = blocks_stop - blocks_start

        self.block_mask = block_mask
        self.blocks_start = blocks_start
        self.blocks_stop = blocks_stop
        self.blocks_shape = blocks_shape
        

def coordlist_to_boolmap(coordlist, bounding_box=None):
    """
    Convert the given list of coordinates (z,y,x) into a 3D bool array.
    
    coordlist: For example, [[0,1,2], [0,1,3], [0,2,0]]
    
    bounding_box: (Optional) Specify the bounding box that corresponds
                  to the region of interest.
                  If not provided, default is to use the smallest bounds
                  that includes the entire coordlist.
    
    Returns: boolmap (3D array, bool), and the bounding_box (start, stop) of the array.
    """
    coordlist = np.asarray(list(coordlist)) # Convert, in case coordlist was a set
    coordlist_min = np.min(coordlist, axis=0)
    coordlist_max = np.max(coordlist, axis=0)
    
    if bounding_box is None:
        start, stop = (coordlist_min, 1+coordlist_max)
    else:
        start, stop = bounding_box
        if (coordlist_min < start).any() or (coordlist_max >= stop).any():
            # Remove the coords that are outside the user's bounding-box of interest
            coordlist = [coord for coord in coordlist if (coord - start >= 0).all() and (coord < stop).all()]
            coordlist = np.array(coordlist)

    shape = stop - start
    coords = coordlist - start
    
    boolmap = np.zeros( shape=shape, dtype=bool )
    boolmap[tuple(coords.transpose())] = 1
    return boolmap, (start, stop)

def block_mask_to_px_mask(block_mask, block_width):
    """
    Given a mask array with block-resolution (each item represents 1 block),
    upscale it to pixel-resolution.
    """
    px_mask_shape = block_width*np.array(block_mask.shape)
    px_mask = np.zeros( px_mask_shape, dtype=np.bool )
    
    # In this 6D array, the first 3 axes address the block index,
    # and the last 3 axes address px within the block
    px_mask_blockwise = view_as_blocks(px_mask, (block_width, block_width, block_width))
    assert px_mask_blockwise.shape[0:3] == block_mask.shape
    assert px_mask_blockwise.shape[3:6] == (block_width, block_width, block_width)
    
    # Now we can broadcast into it from the block mask
    px_mask_blockwise[:] = block_mask[:, :, :, None, None, None]
    return px_mask

def dense_roi_mask_for_subvolume(subvolume, border='default'):
    """
    Return a dense (pixel-level) mask for the given subvolume,
    according to the ROI blocks it lists in its 'intersecting_blocks' member.
    
    border: How much border to incorporate into the mask beyond the subvolume's own bounding box.
            By default, just use the subvolume's own 'border' attribute.
    """
    sv = subvolume
    if border == 'default':
        border = sv.border
    else:
        assert border <= sv.border, \
            "Subvolumes don't store ROI blocks outside of their known border "\
            "region, so I can't produce a mask outside that area."
    
    # subvol bounding box/shape (not block-aligned)
    sv_start_px = np.array((sv.box.z1, sv.box.y1, sv.box.x1)) - border
    sv_stop_px  = np.array((sv.box.z2, sv.box.y2, sv.box.x2)) + border
    sv_shape_px = sv_stop_px - sv_start_px
   
    # subvol bounding box/shape in block coordinates
    sv_start_blocks = sv_start_px // sv.roi_blocksize
    sv_stop_blocks = (sv_stop_px + sv.roi_blocksize-1) // sv.roi_blocksize

    intersecting_block_mask, _ = coordlist_to_boolmap(sv.intersecting_blocks, (sv_start_blocks, sv_stop_blocks))
    intersecting_dense = block_mask_to_px_mask(intersecting_block_mask, sv.roi_blocksize)

    # bounding box of the sv dense coordinates within the block-aligned intersecting_dense
    dense_start = sv_start_px % sv.roi_blocksize
    dense_stop = dense_start + sv_shape_px
    
    # Extract the pixels we want from the (block-aligned) intersecting_dense
    sv_intersecting_dense = intersecting_dense[bb_to_slicing(dense_start, dense_stop)]
    assert sv_intersecting_dense.shape == tuple(sv_shape_px)
    return sv_intersecting_dense

def runlength_encode(coord_list_zyx, assume_sorted=False):
    """
    Given an array of coordinates in the form:
        
        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
        
    Return an array of run-length encodings of the form:
    
        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]
    
    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.
    
    Args:
        coord_list_zyx:
            Array of shape (N,3)
        
        assume_sorted:
            If True, the provided coordinates are assumed to be pre-sorted in Z-Y-X order.
            Otherwise, this function sorts them before the RLEs are computed.
    
    Timing notes:
        The FIB-25 'seven_column_roi' consists of 927971 block indices.
        On that ROI, this function takes 1.65 seconds, but with numba installed,
        it takes 35 ms (after ~400 ms warmup).
        So, JIT speedup is ~45x.
    """
    if len(coord_list_zyx) == 0:
        return np.ndarray( (0,4), np.int64 )

    coord_list_zyx = np.asarray(coord_list_zyx)
    assert coord_list_zyx.ndim == 2
    assert coord_list_zyx.shape[1] == 3
    
    if not assume_sorted:
        sorting_ind = np.lexsort(coord_list_zyx.transpose()[::-1])
        coord_list_zyx = coord_list_zyx[sorting_ind]

    return _runlength_encode(coord_list_zyx)


@jit(nopython=True)
def _runlength_encode(coord_list_zyx):
    """
    Helper function for runlength_encode(), above.
    
    coord_list_zyx:
        Array of shape (N,3), of form [[Z,Y,X], [Z,Y,X], ...],
        pre-sorted in Z-Y-X order.  Duplicates permitted.
    """
    # Numba doesn't allow us to use empty lists at all,
    # so we have to initialize this list with a dummy row,
    # which we'll omit in the return value
    runs = [0,0,0,0]
    
    # Start the first run
    (prev_z, prev_y, prev_x) = current_run_start = coord_list_zyx[0]
    
    for i in range(1, len(coord_list_zyx)):
        (z,y,x) = coord = coord_list_zyx[i]

        # If necessary, end the current run and start a new one
        # (Also, support duplicate coords without breaking the current run.)
        if (z != prev_z) or (y != prev_y) or (x not in (prev_x, 1+prev_x)):
            runs += list(current_run_start) + [prev_x]
            current_run_start = coord

        (prev_z, prev_y, prev_x) = (z,y,x)

    # End the last run
    runs += list(current_run_start) + [prev_x]

    # Return as 2D array
    runs = np.array(runs).reshape((-1,4))
    return runs[1:, :] # omit dummy row (see above)

@jit(nopython=True)
def runlength_decode_from_ranges(rle_array_zyx):
    """
    Given an array of run-length encodings in the form:
        
        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]
        
    Return an array of coordinates of the form:

        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]

    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.
    """
    # Numba doesn't allow us to use empty lists at all,
    # so we have to initialize this list with a dummy row,
    # which we'll omit in the return value
    coords = [0,0,0]
    for i in range(len(rle_array_zyx)):
        (z, y, x1, x2) = rle_array_zyx[i]
        for x in range(x1,x2+1):
            coords += [z,y,x]
    coords = np.array(coords).reshape((-1,3))
    return coords[1:, :] # omit dummy row (see above)


def blockwise_boxes( bounding_box, block_shape ):
    """
    Generator.

    Divides the given global bounding box into blocks and iterate over the block boxes.
    The generated block boxes are aligned to a grid in global coordinates (starting at (0,0,...)).
    Block boxes on the edge of the global bounding box will be clipped so as not to
    extend outside the global bounding box.
    """
    bounding_box = np.asarray(bounding_box, dtype=int)
    block_shape = np.asarray(block_shape)

    # round down, round up
    aligned_start = (bounding_box[0] // block_shape) * block_shape
    aligned_stop = ((bounding_box[1] + block_shape-1) // block_shape) * block_shape
    aligned_shape = aligned_stop - aligned_start

    for box_index in np.ndindex( *(aligned_shape // block_shape) ):
        box_index = np.asarray(box_index)
        box_start = aligned_start + block_shape * box_index
        box_stop  = aligned_start + block_shape * (box_index+1)
        
        # Clip to global bounding box
        box_start = np.maximum( box_start, bounding_box[0] )
        box_stop  = np.minimum( box_stop,  bounding_box[1] )

        yield np.asarray((box_start, box_stop))

def choose_pyramid_depth(bounding_box, top_level_max_dim=512):
    """
    If a 3D volume pyramid were generated to encompass the given bounding box,
    determine how many pyramid levels you would need such that the top
    level of the pyramid is no wider than `top_level_max_dim` in any dimension.
    """
    from numpy import ceil, log2
    bounding_box = np.asarray(bounding_box)
    global_shape = bounding_box[1] - bounding_box[0]

    full_res_max_dim = float(global_shape.max())
    assert full_res_max_dim > 0.0, "Subvolumes encompass no volume!"
    
    depth = int(ceil(log2(full_res_max_dim / top_level_max_dim)))
    return max(depth, 0)


def mask_roi(data, subvolume, border='default'):
    """
    masks data to 0 if outside of ROI stored in subvolume
    
    Note: This function operates on data IN-PLACE
    """
    mask = dense_roi_mask_for_subvolume(subvolume, border)
    assert data.shape == mask.shape
    data[np.logical_not(mask)] = 0
    return None # Emphasize in-place behavior


def nonconsecutive_bincount(label_vol):
    """
    Like np.bincount(), but works well for label volumes with non-consecutive label values.
    Returns two 1D arrays: unique_labels, counts
    Neither array is sorted.
    """
    # Local import because it seems to have (very occasional) trouble importing
    # in the global context during Spark RDD de-serialization
    import pandas as pd
    
    assert isinstance(label_vol, np.ndarray)
    assert np.issubdtype(label_vol.dtype, np.integer)

    counts = pd.Series(np.ravel(label_vol, order='K')).value_counts()
    assert counts.values.dtype == np.int64
    return counts.index, counts.values.view(np.uint64)

def reverse_dict(d):
    rev = { v:k for k,v in d.items() }
    assert len(rev) == len(d), "dict is not reversable: {}".format(d)
    return rev


def select_item(rdd, *indexes):
    """
    Given an RDD of tuples, return an RDD listing the Nth item from each tuple.
    If the tuples are nested, you can provide multiple indexes to drill down to the element you want.
    
    For now, each index must be either 0 or 1.
    
    NOTE: Multiple calls to this function will result in redundant calculations.
          You should probably persist() the rdd before calling this function.
    
    >>> rdd = sc.parallelize([('foo', ('a', 'b')), ('bar', ('c', 'd'))])
    >>> select_item(rdd, 1, 0).collect()
    ['b', 'd']
    """
    for i in indexes:
        if i == 0:
            rdd = rdd.keys()
        else:
            rdd = rdd.values()
    return rdd

def zip_many(*rdds):
    """
    Like RDD.zip(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2

    result = rdds[0].zip(rdds[1])
    rdds = rdds[2:]

    def append_value_to_key(k_v):
        return (k_v[0] + (k_v[1],))

    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.zip(next_rdd).map(append_value_to_key)
    return result

def join_many(*rdds):
    """
    Like RDD.join(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2
    
    result = rdds[0].join(rdds[1])
    rdds = rdds[2:]
    
    def condense_value(k_v):
        k, (v1, v2) = k_v
        return (k, v1 + (v2,))
    
    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.join(next_rdd).map(condense_value, True)
    return result

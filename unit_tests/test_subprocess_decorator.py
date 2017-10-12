import sys
import unittest
from collections import defaultdict
from multiprocessing import TimeoutError

import numpy as np

import logging
logger = logging.getLogger(__name__)

from DVIDSparkServices.subprocess_decorator import execute_in_subprocess, _test_helper, stdout_redirected
from DVIDSparkServices.skeletonize_array import skeletonize_array

from cffi import FFI
ffi = FFI()
ffi.cdef("unsigned int sleep(unsigned int seconds);")
C = ffi.dlopen(None) 

def c_sleep():
    """
    Sleep, but with a C-library function to make sure
    that Python isn't able to interrupt it via SIGTERM.
    """
    C.sleep(3)


class MessageCollector(logging.StreamHandler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_messages = defaultdict(lambda: [])
    
    def emit(self, record):
        msg = self.format(record)
        self.collected_messages[record.levelname].append(msg)


class TestSubprocessDecorator(unittest.TestCase):    
    

    def test_basic(self):
        handler = MessageCollector()
        logging.getLogger().addHandler(handler)
      
        try:        
            result = execute_in_subprocess(1.0, logger)(_test_helper)(1,2,0)
            assert result == 1+2+0, "Wrong result: {}".format(result)
            assert handler.collected_messages['INFO'] == ['1', '0']
            assert handler.collected_messages['ERROR'] == ['2']
              
      
        finally:        
            logging.getLogger().removeHandler(handler)
      

    def test_error(self):
        """
        Generate an exception in the subprocess and verify that it appears in the parent.
        """
        # Too many arguments; should fail
        try:
            _result = execute_in_subprocess(1.0, logger)(_test_helper)(1,2,3,4,5)
        except:
            pass
        else:
            raise RuntimeError("Expected to see an exception in the subprocess.")
      

    def test_timeout(self):
        try:
            _result = execute_in_subprocess(1.0, logger)(_test_helper)(1,2,3.0)
        except TimeoutError:
            pass
        else:
            assert False, "Expected a timeout error."
 

    def test_timeout_in_C_function(self):
        try:
            _result = execute_in_subprocess(1.0, logger)(c_sleep)()
        except TimeoutError:
            pass
        else:
            assert False, "Expected a timeout error."


    def test_timeout_in_skeletonize(self):
        """
        Verify that the subprocess_decorator works to kill the skeletonize function.
        """
        a = np.ones((100,1000,1000), dtype=np.uint8)
        
        try:
            _result = execute_in_subprocess(1.0, logger)(skeletonize_array)(a)
        except TimeoutError:
            pass
        else:
            assert False, "Expected a timeout error."

 
    def test_no_logger(self):
        """
        If no logger is specified to the decorator, then standard out and
        standard error are sent to the parent process stdout/stderr as usual.
        """
        with open('/tmp/captured-stdout.txt', 'w') as f_out, \
             open('/tmp/captured-stderr.txt', 'w') as f_err:
             
            with stdout_redirected( f_out.fileno(), sys.stdout ), \
                 stdout_redirected( f_err.fileno(), sys.stderr ):
    
                result = execute_in_subprocess(1.0)(_test_helper)(1,2,0)
                assert result == 1+2+0, "Wrong result: {}".format(result)
     
        with open('/tmp/captured-stdout.txt', 'r') as f_out, \
             open('/tmp/captured-stderr.txt', 'r') as f_err:
     
            assert f_out.read() == '1\n0'
            assert f_err.read() == '2\n'

if __name__ == "__main__":
    unittest.main()

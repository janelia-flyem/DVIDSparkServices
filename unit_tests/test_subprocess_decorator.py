import unittest
from DVIDSparkServices.subprocess_decorator import execute_in_subprocess, test_helper

class TestSubprocessDecorator(unittest.TestCase):    
    
    def test_basic(self):
        collected_output = []
        result = execute_in_subprocess(stdout_callback=collected_output.append)(test_helper)(1,2,3)
        assert result == 1+2+3, "Wrong result: {}".format(result)

        # Remove timing messages
        collected_output = filter(lambda s: 'took' not in s, collected_output)
        
        assert collected_output == ['1\n', '2\n', '3\n'], \
            "Unexpected output: {}".format(collected_output)

    def test_error(self):
        collected_output = []
        try:
            test_helper_in_subprocess = execute_in_subprocess(stdout_callback=collected_output.append)(test_helper)
            
            # Try giving too many arguments -- should error
            _result = test_helper_in_subprocess(1,2,3,4,5)
        except:
            pass
        else:
            assert False, "Expected an error"
        
        #import sys
        #for line in collected_output:
        #    sys.stdout.write(line)

        # Remove timing messages
        collected_output = filter(lambda s: 'took' not in s, collected_output)

        assert 'Traceback' in collected_output[0], "Expected to see Traceback output"
        assert 'TypeError' in collected_output[-1]

if __name__ == "__main__":
    unittest.main()

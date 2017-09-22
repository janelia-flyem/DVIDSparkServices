import unittest
from DVIDSparkServices.auto_retry import auto_retry

class TestAutoRetry(unittest.TestCase):

    def setUp(self):
        self.COUNTER = 3

    def _check_counter(self):
        self.COUNTER -= 1
        assert self.COUNTER == 0, f'counter is {self.COUNTER}'

    def test_failed_retry(self):
        
        @auto_retry(2, 0.0)
        def should_fail():
            self._check_counter()
        
        try:
            should_fail()
        except AssertionError:
            pass
        else:
            assert False, "should_fail() didn't fail!"

    def test_successful_retry(self):        

        @auto_retry(3, 0.0)
        def should_succeed():
            self._check_counter()
    
        try:
            should_succeed()
        except AssertionError:
            assert False, "should_succeed() didn't succeed!"

    def test_failed_with_predicate(self):
        def predicate(ex):
            return '1' in ex.args[0]

        @auto_retry(2, 0.0, predicate=predicate)
        def should_fail():
            self._check_counter()
        
        try:
            should_fail()
        except AssertionError:
            pass
        else:
            assert False, "should_fail() didn't fail!"

    def test_success_with_predicate(self):
        def predicate(ex):
            return '1' in ex.args[0] or '2' in ex.args[0]
        
        @auto_retry(3, 0.0, predicate=predicate )
        def should_succeed():
            self._check_counter()
        
        try:
            should_succeed()
        except AssertionError:
            assert False, "should_succeed() didn't succeed!"


if __name__ == "__main__":
    unittest.main()

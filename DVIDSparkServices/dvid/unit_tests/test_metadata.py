import unittest
from libdvid import DVIDNodeService, DVIDServerService

from DVIDSparkServices.dvid.metadata import is_dvidversion, is_datainstance, dataInstance 

dvidserver = "http://127.0.0.1:8000"

class Testmetadata(unittest.TestCase):
    """Tests dvid metadata.  Only works if 127.0.0.1:8000 contains DVID.
    """
   
    def test_isdvidversion(self):
        """Tests is_dvidversion function.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        self.assertTrue(is_dvidversion(dvidserver, uuid))
        self.assertFalse(is_dvidversion(dvidserver, uuid + "JUNK"))

    def test_isdatainstance(self):
        """Tests is_datainstance function.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_labelblk("labels")

        self.assertTrue(is_datainstance(dvidserver, uuid, "labels"))
        self.assertFalse(is_datainstance(dvidserver, uuid, "labels2"))

    def test_dataInstance(self):
        """Tests is_dataInstance class.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_labelblk("labels")
        ns.create_grayscale8("gray")
        ns.create_keyvalue("kv")

        try:
            temp = dataInstance(dvidserver, uuid, "blah")
        except ValueError:
            # correct caught error
            self.assertTrue(True)
            
        labels = dataInstance(dvidserver, uuid, "labels")
        gray = dataInstance(dvidserver, uuid, "gray")
        kv = dataInstance(dvidserver, uuid, "kv")

        self.assertTrue(labels.is_array())
        self.assertTrue(labels.is_labels())
        
        self.assertTrue(gray.is_array())
        self.assertFalse(gray.is_labels())

        self.assertFalse(kv.is_array())
        self.assertFalse(kv.is_labels())

if __name__ == "main":
    unittest.main()

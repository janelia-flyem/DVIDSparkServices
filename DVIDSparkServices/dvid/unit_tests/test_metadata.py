import unittest
import json
from libdvid import DVIDNodeService, DVIDServerService, ConnectionMethod, DVIDConnection

from DVIDSparkServices.dvid.metadata import is_dvidversion, is_datainstance, DataInstance, set_sync, has_sync, get_blocksize, create_rawarray8, create_labelarray, Compression 

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

    def test_sycns(self):
        """Test sync check and setting a sync.
        """
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
       
        create_labelarray(dvidserver, uuid, "labels")
        
        # check if labels is listening to labels2
        self.assertFalse(has_sync(dvidserver, uuid, "labels", "bodies"))

        # create labelvol and sync to it
        conn = DVIDConnection(dvidserver) 

        endpoint = "/repo/" + uuid + "/instance"
        data = {"typename": "labelvol", "dataname": "bodies"}
        conn.make_request(endpoint, ConnectionMethod.POST, json.dumps(data))

        set_sync(dvidserver, uuid, "labels", "bodies")
        self.assertTrue(has_sync(dvidserver, uuid, "labels", "bodies"))

    def test_create_rawarray8(self):
        """Test creation of rawarray and block size fetch.
        """
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
       
        create_rawarray8(dvidserver, uuid, "gray", (32,16,14), Compression.JPEG)
        blocksize = get_blocksize(dvidserver, uuid, "gray") 
        self.assertEqual(blocksize, (32,16,14))


    def test_create_labelarray(self):
        """Test creation of labelarray and block size fetch
        """
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
       
        create_labelarray(dvidserver, uuid, "labels")
        blocksize = get_blocksize(dvidserver, uuid, "labels") 
        self.assertEqual(blocksize, (64,64,64))


    def test_DataInstance(self):
        """Tests DataInstance class.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_labelblk("labels")
        ns.create_grayscale8("gray")
        ns.create_keyvalue("kv")

        try:
            temp = DataInstance(dvidserver, uuid, "blah")
        except ValueError:
            # correct caught error
            self.assertTrue(True)
            
        labels = DataInstance(dvidserver, uuid, "labels")
        gray = DataInstance(dvidserver, uuid, "gray")
        kv = DataInstance(dvidserver, uuid, "kv")

        self.assertTrue(labels.is_array())
        self.assertTrue(labels.is_labels())
        
        self.assertTrue(gray.is_array())
        self.assertFalse(gray.is_labels())

        self.assertFalse(kv.is_array())
        self.assertFalse(kv.is_labels())

if __name__ == "main":
    unittest.main()

"""Implements agglomerations of supervoxels using Segmentor workflow and neuroproof.
"""
from __future__ import print_function, absolute_import
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

def neuroproof_agglomerate(grayscale, predictions, supervoxels, classifier, threshold = 0.20, mitochannel = 2):
    """Main agglomeration function

   Args:
        grayscale = 3D uing8 (z,y,x) -- Not used.
        predictions = 4D float32 numpy label array (z, y, x, ch) 
        supervoxels = 3D uint32 numpy label array (z,y,x) 
        classifier = file location or DVID (assume to be xml unless .h5 is explict in name)
        threshold = threshold (default = 0.20)
        mitochannel = prediction channel for mito (default 2) (empty means no mito mode)
    
    Returns:
        segmentation = 3D numpy label array (z,y,x)
    """

    print("neuroproof_agglomerate(): Starting with label data: dtype={}, shape={}".format(str(supervoxels.dtype), supervoxels.shape))


    import numpy
    # return immediately if no segmentation
    if len(numpy.unique(supervoxels)) <= 1:
        return supervoxels


    #from neuroproof import Classifier, Agglomeration
    from neuroproof import Agglomeration
    import os

    # verify channels
    assert predictions.ndim == 4
    z,y,x,nch = predictions.shape

    if nch > 2:
        # make sure mito is in the second channel
        predictions[[[[2, mitochannel]]]] = predictions[[[[mitochannel, mitochannel]]]] 

    pathname = str(classifier["path"])
    tempfilehold = None
    tclassfile = ""

    # write classifier to temporary file if stored on DVID
    if "dvid-server" in classifier:
        # allow user to specify any server and version for the data
        dvidserver = classifier["dvid-server"]
        uuid = classifier["uuid"]

        # extract file and store into temporary location
        node_service = retrieve_node_service(str(dvidserver), str(uuid))

        name_key = pathname.split('/')
        classfile = node_service.get(name_key[0], name_key[1])

        # create temp file
        import tempfile
        tempfilehold = tempfile.NamedTemporaryFile(delete=False)
       
        # open file and write data
        with open(tempfilehold.name, 'w') as fout:
            fout.write(classfile)

        # move temporary file to have the same extension as provided file
        if pathname.endswith('.h5'):
            tclassfile = tempfilehold.name + ".h5"
        else:
            tclassfile = tempfilehold.name + ".xml"
        os.rename(tempfilehold.name, tclassfile)

    else:
        # just read from directory
        tclassfile = pathname
        

    # load classifier from file
    #classifier = loadClassifier(tclassfile)

    # run agglomeration (supervoxels must be 32 uint and predicitons must be float32)
    segmentation = Agglomeration.agglomerate(supervoxels.astype(numpy.uint32), predictions.astype(numpy.float32), tclassfile, threshold)

    if tempfilehold is not None:
        os.remove(tclassfile)

    return segmentation



"""This module defines routines to create 3D image volumes from 2D files.
"""

from __future__ import absolute_import
from .volumeSrc import volumeSrc
from . import partitionSchema
from PIL import Image
import numpy as np


class imagefileSrc(volumeSrc):
    """Iterator class provides interface to image file data.

    This class reads a series a files from disk.  It supports standard image
    format such that can be handled python PIL,  It also supports files stored
    in google storage bucket (prefixed by gs://).

    Format Examples:
        
        directory (must end with '/' or '\') : fileprefix: "/images/"
        wildcard images: fileprefix: "/images/*.png"
        formatted string: fileprefix: "/images/img%05.png" minmaxplane(50, 5000)
        google bucket: fileprefix: "gs://FILE%dFORMAT" minmaxplane: (0, 100)

    Note:
        If an minmaxplane is not provided with a formatted string, the files
        are ordered lexicographically.  Google buckets currently only supports
        formatted strings with minmaxplane specified.  To use google buckets as
        an input source, the application must have application wide read permission
        into Google storage.  This can be done, for instance, by setting the
        environment variable GOOGLE_APPLICATION_CREDENTIALS to point to the
        proper credentials.
    """

    def __init__(self, part_schema, fileprefix, minmaxplane = None,
            offset=partitionSchema.VolumeOffset(0,0,0), spark_context = None, dtype=None):
        """Initialization.    
        
        Args:
            part_schema (partitionSchema): describes the image volume and its partitioning rules
            fileprefix (string): files path formatted as directory, wildcard, or formatted string
            minmaxplane (tuple(int,int)): delimits image slices with formatted string
            spark_context (sparkconf): determines if spark is used for fetching data
            dtype (numpy.dtype): specifies size of voxel (numpy array's used internally)
        """

        super(imagefileSrc, self).__init__(part_schema)
        self.filelist = []
        self.filepos = 0
        self.spark_context = spark_context
        self.dtype = dtype

        if minmaxplane is None:
            # extract files in directory or matching the specified wildcards
            import glob
            
            if fileprefix.endswith('/') or fileprefix.endswith('\\'):
                fileprefix += '*'
            self.filelist = glob.glob(fileprefix)
            self.filelist.sort()
        else:
            # minplane defines which image is fetched but the global offset
            # is determined by the partition schema
            minplane, maxplane = minmaxplane 
            self.slicenum = minplane

            if '%' in fileprefix: 
                for slicenum in range(minplane, maxplane+1):
                    self.filelist.append(fileprefix % slicenum)
            elif '{' in fileprefix:
                for slicenum in range(minplane, maxplane+1):
                    self.filelist.append(fileprefix.format(slicenum))
            else:
                raise RuntimeError(f"Unrecognized format string for fileprefix: {fileprefix}")

        # find iteration start and finish plane
        self.offset = offset
        offsetz = self.offset.z
        self.start_slice = offsetz
        self.curr_slice = self.start_slice
        self.end_slice = offsetz+len(self.filelist)-1

        # find proper iteration size
        # if nothing is specified default to one (user should use
        # get_volume if they do not want iteration)
        self.iteration_size = 1
        if part_schema.get_partdims().zsize != 0:
            zshift = offsetz % part_schema.get_partdims().zsize
            self.curr_slice = self.start_slice - zshift 
            self.iteration_size = part_schema.get_partdims().zsize

    def __iter__(self):
        """Defines iterable type.
        """
        return self

    def _retrieve_images(self, start, size):
        start_slice = self.start_slice
        end_slice = self.end_slice
        dtype = self.dtype
        filelist = self.filelist
        offset = self.offset

        def img2npy(slicenum):
            # get address for this slice
            partition = partitionSchema.volumePartition(slicenum, partitionSchema.VolumeOffset(slicenum, offset.y, offset.x))
            # no file exists for the specified range
            if slicenum < start_slice or slicenum > end_slice:
                return partition, None 
            filename = filelist[slicenum-start_slice] 
            # check for google storage prefix
            gbucketsrc = False
            if filename.startswith("gs://"):
                gbucketsrc = True 
            try:
                img = None
                if not gbucketsrc:
                    img = Image.open(filename)
                else:
                    from gcloud import storage
                    from io import BytesIO
                    
                    # strip gs:// and extract bucket and object name
                    gspath = filename[5:]
                    bucketpath = gspath.split('/')
                    gbucketname = bucketpath[0]
                    filename = '/'.join(bucketpath[1:])
                    client = storage.Client()
                    gbucket = client.get_bucket(gbucketname)
                    gblob = gbucket.get_blob(filename)
                    
                    # write to bytes which implements file interface
                    gblobfile = BytesIO()
                    gblob.download_to_file(gblobfile)
                    gblobfile.seek(0)
                    img = Image.open(gblobfile)
             
                img_array = None
                if dtype is None:
                    # use default image datatype
                    img_array = np.array(img)
                else:
                    img_array = np.array(img, dtype=dtype)
                return partition, img_array
            except Exception as e:
                # just return a blank slice -- will be handled downstream
                return partition, None 

        if self.spark_context is not None:
            # requires spark application
            imgs = self.spark_context.parallelize(list(range(start, start+size)), size)
            return imgs.map(img2npy)        
        else:
            # produces an array of 2D numpy images
            img_list = []
            for slicenum in range(start, start+size):
                img_list.append(img2npy(slicenum))

            return img_list

    def __next__(self):
        """Iterates partitions specified in the partitionSchema.
        """
        if self.curr_slice > self.end_slice:
            raise StopIteration()

        # RDD or array of images
        images = self._retrieve_images(self.curr_slice, self.iteration_size)
        self.curr_slice += self.iteration_size

        # partition data
        return self.part_schema.partition_data(images)

    # Python 2
    next = __next__

    def extract_volume(self):
        """Retrieve entire volume as numpy array or RDD.
        """
        
        # RDD or array of images
        images = self._retrieve_images(self.curr_slice, self.end_slice - self.curr_slice + 1)

        # partition data (empty partitioner data will just return one numpy array)

        return self.part_schema.partition_data(images)





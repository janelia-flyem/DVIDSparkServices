class Segmentor(object):
    # determines what pixels are used as seeds (0-255 8-bit range)
    SEED_THRES = 10 
   
    # the size of the seed needed
    SEED_SIZE_THRES = 5

    def __init__(self, context, config):
        self.context = context

    # simple, default seeded watershed-based segmentation
    def segment(self, gray_chunks):
        from scipy.ndimage import label
        from scipy.ndimage.morphology import binary_closing
        from numpy import bincount 
        from skimage import morphology as skmorph

        def _segment(gray_chunks):
            (key, subvolume, gray) = gray_chunks

            # extract seed mask from grayscale
            seedmask = gray[gray <= SEED_THRES]

            # closing operation to clean up seed mask
            seedmask_closed = binary_closing(seedmask, iterations=2)

            # split into connected components
            seeds = label(seedmask_closed)[0]

            # filter small connected components
            component_sizes = bincount(seeds.ravel())
            small_components = component_sizes < SEED_SIZE_THRES
            small_locations = small_components[seeds]
            seeds[small_locations] = 0

            # compute watershed
            supervoxels = skmorph.watershed(gray, seeds)
            max_id = supervoxels.max()
            supervoxels_compressed = CompressedNumpyArray(supervoxels) 
            subvolume.set_max_id(max_id)

            return (key, subvolume, supervoxels_compressed)

        # preserver partitioner
        return gray_chunks.mapValues(_segment): 

    # label volumes to label volumes remapped, preserves partitioner 
    def stitch(self, label_chunks):
        # return all subvolumes back to the driver
        # create offset map (substack id => offset) and broadcast
        subvolumes = label_chunks.map(lambda x: return x[1]).collect()
        offsets = {}
        offset = 0
        for subvolume in subvolumes:
            offsets[subvolume.roi_id] = offset
            offset += subvolume.max_id
        subvolume_offsets = self.context.sc.broadcast(offsets)

        # ?!
        def extract_boundaries:
            pass

        # return compressed boundaries (id1-id2, boundary)
        mapped_boundaries = label_chunks.flatMap(extract_boundaries) 

        # shuffle the hopefully smallish boundaries into their proper spot
        # groupby is not a big deal here since same keys will not be in the same partition
        grouped_boundaries = mapped_boundaries.groupByKey()

        # ?!
        def stither:
            pass

        # map from grouped boundary to substack id, mappings
        subvolume_mappings = grouped_boundaries.flatMap(sticher)

        # ?!
        def relabel:
            pass

        # grouping operations shouldn't be too expensive since
        # label chunk partitioner was preserved
        return label_chunks.group(subvolume_mappings).mapValues(relabel)



class Segmentor(object):
    # determines what pixels are used as seeds (0-255 8-bit range)
    SEED_THRES = 10 
   
    # the size of the seed needed
    SEED_SIZE_THRES = 5

    def __init__(self, context, config):
        self.context = context
        mode = config["options"]["stitch-algorithm"]
        if mode == "none":
            self.stitch_mode = 0
        elif mode == "conservative":
            self.stitch_mode = 1
        elif mode == "medium":
            self.stitch_mode = 2
        elif mode == "aggressive":
            self.stitch_mode = 3
        else:
            raise Exception("Invalid stitch mode specified")


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

            return (key, (subvolume, supervoxels_compressed))

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

        # (key, subvolume, label chunk)=> (new key, (subvolume, boundary))
        def extract_boundaries(key_labels):
            # compute overlap -- assume first point is less than second
            def intersects(pt1, pt2, pt1_2, pt2_2):
                if pt1 > pt2:
                    raise Exception("point 1 greater than point 2")
                if pt1_2 > pt2_2:
                    raise Exception("point 1 greater than point 2")

                val1 = max(pt1, pt1_2)
                val2 = min(pt2, pt2_2)
                size = val2-val1
                npt1 = val1 - pt1 
                npt1_2 = val1 - pt1_2

                return npt1, npt1+size, npt1_2, npt1_2+size

            import numpy
            from DVIDSparkServices.sparkdvid import CompressedNumpyArray

            oldkey, (subvolume, labelsc) = key_labels
            labels = labelsc.decompress()

            boundary_array = []
            
            # iterate through all ROI partners
            for partner in subvolume.local_regions:
                key1 = subvolume.roi_id
                key2 = partner[0]
                roi2 = partner[1]
                if key2 < key1:
                    key1, key2 = key2, key1
                
                # create key for boundary pair
                newkey = (key1, key2)

                # crop volume to overlap
                offx1, offx2, offx1_2, offx2_2 = 
                            intersects(
                                subvolume.roi.x1-subvolume.border,
                                subvolume.roi.x2+subvolume.border,
                                roi2.x1-subvolume.border,
                                roi2.x2+subvolume.border
                            )
                offy1, offy2, offy1_2, offy2_2 = 
                            intersects(
                                subvolume.roi.y1-subvolume.border,
                                subvolume.roi.y2+subvolume.border,
                                roi2.y1-subvolume.border,
                                roi2.y2+subvolume.border
                            )
                offz1, offz2, offz1_2, offz2_2 = 
                            intersects(
                                subvolume.roi.z1-subvolume.border,
                                subvolume.roi.z2+subvolume.border,
                                roi2.z1-subvolume.border,
                                roi2.z2+subvolume.border
                            )
                            
                labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2]

                labels_cropped_c = CompressedNumpyArray(labels_cropped)
                # add to flat map
                boundary_array.append((new_key, (subvolume, labels_cropped_c)))

            return boundary_array


        # return compressed boundaries (id1-id2, boundary)
        mapped_boundaries = label_chunks.flatMap(extract_boundaries) 

        # shuffle the hopefully smallish boundaries into their proper spot
        # groupby is not a big deal here since same keys will not be in the same partition
        grouped_boundaries = mapped_boundaries.groupByKey()

        stitch_mode = self.stich_mode

        # mappings to one partition (larger/second id keeps orig labels)
        # (new key, list<2>(subvolume, boundary compressed)) =>
        # (key, (subvolume, mappings))
        def stitcher(key_boundary):
            key, (boundary_list) = key_boundary

            # should be only two values
            if len(boundary_list) != 2:
                raise Exception("Expects exactly two subvolumes per boundary")

            # order subvolume regions (they should be the same shape)
            subvolume1, boundary1_c = boundary_list[0]
            subvolume2, boundary2_c = boundary_list[1]

            if subvolume1.roi_id > subvolume2.roi_id:
                subvolume1, subvolume2 = subvolume2, subvolume1
                boundary1_c, boundary2_c = boundary2_c, boundary1_c

            boundary1 = boundary1_c.decompress()
            boundary2 = boundary2_c.decompress()

            if boundary1.shape != boundary2.shape:
                raise Exception("Extracted boundaries are different shapes")
            
            # determine list of bodies in play
            z2, y2, x2 = boundary1.shape
            z1 = y1 = x1 = 0 

            # determine which interface there is touching between subvolumes 
            if subvolume1.touches(subvolume1.roi.x1, subvolume1.roi.x2,
                                subvolume2.roi.x1, subvolume2.roi.x2):
                x1 = x2/2 
                x2 = x1 + 1
            if subvolume1.touches(subvolume1.roi.y1, subvolume1.roi.y2,
                                subvolume2.roi.y1, subvolume2.roi.y2):
                y1 = y2/2 
                y2 = y1 + 1
            
            if subvolume1.touches(subvolume1.roi.z1, subvolume1.roi.z2,
                                subvolume2.roi.z1, subvolume2.roi.z2):
                z1 = z2/2 
                z2 = z1 + 1

            eligible_bodies = set(numpy.unique(boundary2[z1:z2, y1:y2, x1:x2]))
            body2body = {}

            label2_bodies = numpy.unique(boundary2)

            # 0 is off,
            # 1 is very conservative (high percentages and no bridging),
            # 2 is less conservative (no bridging),
            # 3 is the most liberal (some bridging allowed if overlap
            # greater than X and overlap threshold)
            hard_lb = 50
            liberal_lb = 1000
            conservative_overlap = 0.90

            if stitch_mode > 0:
                for body in label2_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    if body1 not in body2body[body2]:
                        body2body[body2][body1] = 0
                    body2body[body2][body1] += 1


            # create merge list 
            merge_list = []
            mutual_list = {}
            retired_list = set()

            small_overlap_prune = 0
            conservative_prune = 0
            aggressive_add = 0
            not_mutual = 0

            for body2, bodydict in body2body.items():
                if body2 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body1, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body1
                            max_val = val
                    if bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(bodysave), int(body2)])
                        # do not add
                        retired_list.add((int(bodysave), int(body2))) 
                        aggressive_add += 1
                    else:
                        if int(bodysave) not in mutual_list:
                            mutual_list[int(bodysave)] = {}
                        mutual_list[int(bodysave)][int(body2)] = max_val
                       

            eligible_bodies = set(numpy.unique(boundary1[z1:z2, y1:y2, x1:x2]))
            body2body = {}
            
            if stitch_mode > 0:
                label1_bodies = numpy.unique(boundary1)
                for body in label1_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    if body2 not in body2body[body1]:
                        body2body[body1][body2] = 0
                    body2body[body1][body2] += 1
            
            # add to merge list 
            for body1, bodydict in body2body.items():
                if body1 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body2, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body2
                            max_val = val

                    if (int(body1), int(bodysave)) in retired_list:
                        # already in list
                        pass
                    elif bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(body1), int(bodysave)])
                        aggressive_add += 1
                    elif int(body1) in mutual_list:
                        partners = mutual_list[int(body1)]
                        if int(bodysave) in partners:
                            merge_list.append([int(body1), int(bodysave)])
                        else:
                            not_mutual += 1
                    else:
                        not_mutual += 1
            
            # handle offsets in mergelist
            offset1 = subvolume_offsets[subvolume1.roi_id] 
            offset2 = subvolume_offsets[subvolume2.roi_id] 
            for merger in mergelist:
                merger[0] = merger[0]+offset1
                merger[1] = merger[1]+offset1

            # return id and mappings, only relevant for stack one
            return (subvolume1.substack_id, merge_list)

        # key, mapping1; key mapping2 => key, mapping1+mapping2
        def reduce_mappings(b1, b2):
            b1.extend(b2)
            return b1

        # map from grouped boundary to substack id, mappings
        subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

        # reconcile all the mappings by sending them to the driver
        # (not a lot of data and compression will help but not sure if there is a better way)
        merge_list = []
        all_mappings = subvolume_mappings.collect()
        for (substack_id, mapping) in all_mappings:
            merge_list.extend(mapping)


        # make a body2body map
        body1body2 = {}
        body2body1 = {}
        for merger in merge_list:
            # body1 -> body2
            body1 = merger[0]
            if merger[0] in body1body2:
                body1 = body1body2[merger[0]]
            body2 = merger[1]
            if merger[1] in body1body2:
                body2 = body1body2[merger[1]]

            if body2 not in body2body1:
                body2body1[body2] = set()
            
            # add body1 to body2 map
            body2body1[body2].add(body1)
            # add body1 -> body2 mapping
            body1body2[body1] = body2

            if body1 in body2body1:
                for tbody in body2body1[body1]:
                    body2body1[body2].add(tbody)
                    body1body2[tbody] = body2

        body2body = zip(body1body2.keys(), body1body2.values())
        
        # potentially costly broadcast
        # (possible to split into substack to make more efficient but compression should help)
        master_merge_list = self.context.sc.broadcast(merge_list)

        # use offset and mappings to relabel volume
        def relabel(key_label_mapping):
            key, (subvolume, label_chunk_c) = key_label_mapping
            labels = label_chunk_c.decompress()

            # grab broadcast offset
            offset = subvolume_offsets[subvolume.roi_id]

            labels = labels + offset 
            # make sure 0 is 0
            labels[labels == offset] = 0

            # create default map 
            mapping_col = numpy.unique(labels)
            label_mappings = dict(zip(mapping_col, mapping_col))
            
            # create maps from merge list
            for mapping in master_merge_list:
                if mapping[0] in label_mappings:
                    label_mappings[mapping[0]] = mapping[1]

            # apply maps
            vectorized_relabel = numpy.frompyfunc(label_mappings.__getitem__, 1, 1)
            labels = vectorized_relabel(labels).astype(numpy.uint64)
       
            return (key, CompressedNumpyArray(labels)) 

        # just map values with broadcast map
        # Potential TODO: consider fast join with partitioned map (not broadcast)
        return label_chunks.mapValues(relabel)



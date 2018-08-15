import unittest
from functools import partial

import numpy as np

from neuclease.util import extract_subvol, box_intersection, Grid
from DVIDSparkServices.io_util.brick import ( Brick, generate_bricks_from_volume_source,
                                              realign_bricks_to_new_grid, split_brick, assemble_brick_fragments,
                                              pad_brick_data_from_volume_source )

class TestBrickFunctions(unittest.TestCase):

    def test_generate_bricks(self):
        grid = Grid( (10,20), (12,3) )
        bounding_box = np.array([(15,30), (95,290)])
        volume = np.random.randint(0,10, (100,300) )

        bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume) )

        bricks = list(bricks)
        assert len(bricks) == 9 * 14
        
        for brick in bricks:
            assert isinstance( brick, Brick )
            assert brick.logical_box.shape == (2,2)
            assert brick.physical_box.shape == (2,2)

            # logical_box must be exactly one block
            assert ((brick.logical_box[1] - brick.logical_box[0]) == grid.block_shape).all()
            
            # Must be grid-aligned
            assert ((brick.logical_box - grid.offset) % grid.block_shape == 0).all()
            
            # Must not exceed bounding box
            assert (brick.physical_box == box_intersection( brick.logical_box, bounding_box )).all()
            
            # Volume shape must match
            assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
            
            # Volume data must match
            assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()


    def test_split_brick(self):
        grid = Grid( (10,20), (12,3) )
        volume = np.random.randint(0,10, (100,300) )
        
        # Test with the first brick in the grid
        physical_start = np.array(grid.offset)
        logical_start = physical_start // grid.block_shape * grid.block_shape
        logical_stop = logical_start + grid.block_shape
        
        physical_stop = logical_stop # Not always true, but happens to be true in this case.
        
        logical_box = np.array([logical_start, logical_stop])
        physical_box = np.array([physical_start, physical_stop])
        
        assert (logical_box == [(10,0), (20,20)]).all()
        assert (physical_box == [(12,3), (20,20)]).all()

        original_brick = Brick( logical_box, physical_box, extract_subvol(volume, physical_box) )

        # New grid scheme
        new_grid = Grid((2,10), (0,0))
        boxes_and_fragments = split_brick(new_grid, original_brick)
        boxes, fragments = list(zip(*boxes_and_fragments))

        assert boxes == ( # ((10, 0), (14, 10)),  # <--- Not present. These new boxes intersect with the original logical_box,
                          # ((10, 10), (14, 20)), # <--- but there is no physical data for them in the original brick.
                          ((12, 0), (14, 10)),
                          ((12, 10), (14, 20)),
                          ((14, 0), (16, 10)),
                          ((14, 10), (16, 20)),
                          ((16, 0), (18, 10)),
                          ((16, 10), (18, 20)),
                          ((18, 0), (20, 10)),
                          ((18, 10), (20, 20)) )
        
        for frag in fragments:
            assert (frag.volume == extract_subvol(volume, frag.physical_box)).all()


    def test_assemble_brick_fragments(self):
        volume = np.random.randint(0,10, (100,300) )
        
        logical_box = np.array( [(10, 20), (20, 120)] )

        # Omit the first and last boxes, to prove that the final
        # physical box ends up smaller than the logical box.
        
        # box_0 = np.array( [(10,20), (20,40)] )
        box_1 = np.array( [(10,40), (20,60)] )
        box_2 = np.array( [(10,60), (20,80)] )
        box_3 = np.array( [(10,80), (20,100)] )
        # box_4 = np.array( [(10,100), (20,120)] )

        # frag_0 = Brick( logical_box, box_0, extract_subvol(volume, box_0) ) # omit
        frag_1 = Brick( logical_box, box_1, extract_subvol(volume, box_1) )
        frag_2 = Brick( logical_box, box_2, extract_subvol(volume, box_2) )
        frag_3 = Brick( logical_box, box_3, extract_subvol(volume, box_3) )
        # frag_4 = Brick( logical_box, box_4, extract_subvol(volume, box_4) ) # omit

        assembled_brick = assemble_brick_fragments( [frag_1, frag_2, frag_3] )
        assert (assembled_brick.logical_box == logical_box).all()
        assert (assembled_brick.physical_box == [box_1[0], box_3[1]] ).all()
        
        physical_shape = assembled_brick.physical_box[1] - assembled_brick.physical_box[0]
        assert (assembled_brick.volume.shape == physical_shape).all()
        assert (assembled_brick.volume == extract_subvol(volume, assembled_brick.physical_box)).all()
        

    def test_realign_bricks_to_new_grid(self):
        grid = Grid( (10,20), (12,3) )
        bounding_box = np.array([(15,30), (95,290)])
        volume = np.random.randint(0,10, (100,300) )

        original_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume) )

        new_grid = Grid((20,10), (0,0))
        boxes_and_bricks = realign_bricks_to_new_grid(new_grid, original_bricks)

        new_logical_boxes, new_bricks = list(zip(*boxes_and_bricks))

        assert len(new_bricks) == 5 * 26 # from (0,30) -> (100,290)
        
        for logical_box, brick in zip(new_logical_boxes, new_bricks):
            assert isinstance( brick, Brick )
            assert (brick.logical_box == logical_box).all()

            # logical_box must be exactly one block
            assert ((brick.logical_box[1] - brick.logical_box[0]) == new_grid.block_shape).all()
            
            # Must be grid-aligned
            assert ((brick.logical_box - new_grid.offset) % new_grid.block_shape == 0).all()
            
            # Must not exceed bounding box
            assert (brick.physical_box == box_intersection( brick.logical_box, bounding_box )).all()
            
            # Volume shape must match
            assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
            
            # Volume data must match
            assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()


    def test_pad_brick_data_from_volume_source(self):
        source_volume = np.random.randint(0,10, (100,300) )
        logical_box = [(1,0), (11,20)]
        physical_box = [(3,8), (7, 13)]
        brick = Brick( logical_box, physical_box, extract_subvol(source_volume, physical_box) )
        
        padding_grid = Grid( (5,5), offset=(1,0) )
        padded_brick = pad_brick_data_from_volume_source( padding_grid, partial(extract_subvol, source_volume), brick )
        
        assert (padded_brick.logical_box == brick.logical_box).all()
        assert (padded_brick.physical_box == [(1,5), (11, 15)]).all()
        assert (padded_brick.volume == extract_subvol(source_volume, padded_brick.physical_box)).all()


    def test_pad_brick_data_from_volume_source_NO_PADDING_NEEDED(self):
        source_volume = np.random.randint(0,10, (100,300) )
        logical_box = [(1,0), (11,20)]
        physical_box = [(6,10), (11, 15)]
        brick = Brick( logical_box, physical_box, extract_subvol(source_volume, physical_box) )
        
        padding_grid = Grid( (5,5), offset=(1,0) )
        padded_brick = pad_brick_data_from_volume_source( padding_grid, partial(extract_subvol, source_volume), brick )

        assert padded_brick is brick, "Expected to get the same brick back."

class TestBrickFunctionsWithHalo(unittest.TestCase):

    def test_generate_bricks(self):
        halo = 1
        halo_shape = np.array([1,1])
        grid = Grid( (10,20), (12,3), halo )
        bounding_box = np.array([(15,30), (95,290)])
        volume = np.random.randint(0,10, (100,300) )

        bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume) )

        bricks = list(bricks)
        assert len(bricks) == 9 * 14
        
        for brick in bricks:
            assert isinstance( brick, Brick )
            assert brick.logical_box.shape == (2,2)
            assert brick.physical_box.shape == (2,2)

            # logical_box must be exactly one block
            assert ((brick.logical_box[1] - brick.logical_box[0]) == grid.block_shape).all()
            
            # Must be grid-aligned
            assert ((brick.logical_box - grid.offset) % grid.block_shape == 0).all()
            
            # Physical == logical+halo, except for bounding-box edges
            assert (brick.physical_box == box_intersection( brick.logical_box + (-halo_shape, halo_shape), bounding_box )).all()
            
            # Volume shape must match
            assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
            
            # Volume data must match
            assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()

    def test_split_brick(self):
        halo = 1
        grid = Grid( (10,20), (12,3), halo )
        volume = np.random.randint(0,10, (100,300) )
        
        # Test with the first brick in the grid
        physical_start = np.array(grid.offset)
        logical_start = physical_start // grid.block_shape * grid.block_shape
        logical_stop = logical_start + grid.block_shape
        
        physical_stop = logical_stop+halo # Not always true, but happens to be true in this case.
        
        logical_box = np.array([logical_start, logical_stop])
        physical_box = np.array([physical_start, physical_stop])
        
        assert (logical_box == [(10,0), (20,20)]).all()
        assert (physical_box == [(12,3), (21,21)]).all()

        original_brick = Brick( logical_box, physical_box, extract_subvol(volume, physical_box) )

        # New grid scheme
        new_grid = Grid((2,10), (0,0))
        
        try:
            _boxes_and_fragments = split_brick(new_grid, original_brick)
        except AssertionError:
            pass # Expected failure: Forbidden to split bricks that have a halo
        else:
            assert False, "Did not encounter the expected assertion.  split_brick() should fail for bricks that have a halo."


    def test_realign_bricks_to_new_grid(self):
        grid = Grid( (10,20), (12,3) )
        bounding_box = np.array([(15,30), (95,290)])
        volume = np.random.randint(0,10, (100,300) )

        original_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume) )

        halo = 1
        halo_shape = np.array([1,1])
        new_grid = Grid((20,10), (0,0), halo)
        boxes_and_bricks = realign_bricks_to_new_grid(new_grid, original_bricks)

        new_logical_boxes, new_bricks = list(zip(*boxes_and_bricks))

        assert len(new_bricks) == 5 * 26, f"{len(new_bricks)}" # from (0,30) -> (100,290)
        
        for logical_box, brick in zip(new_logical_boxes, new_bricks):
            assert isinstance( brick, Brick ), f"Got {type(brick)}"
            assert (brick.logical_box == logical_box).all()

            # logical_box must be exactly one block
            assert ((brick.logical_box[1] - brick.logical_box[0]) == new_grid.block_shape).all()
            
            # Must be grid-aligned
            assert ((brick.logical_box - new_grid.offset) % new_grid.block_shape == 0).all()
            
            # Should match logical_box+halo, except for edges
            assert (brick.physical_box == box_intersection( brick.logical_box + (-halo_shape, halo_shape), bounding_box )).all()
            
            # Volume shape must match
            assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
            
            # Volume data must match
            assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()


if __name__ == "__main__":
    unittest.main()

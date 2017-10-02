import os
import json
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np
import snappy

# DEBUG 'logging' (Doesn't actually use logging module)
#import httplib2
#httplib2.debuglevel = 1

BRAINMAPS_API_VERSION = 'v1beta2'
BRAINMAPS_BASE_URL = f'https://brainmaps.googleapis.com/{BRAINMAPS_API_VERSION}'

def fetch_json(http, url):
    response, content = http.request(url)
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content}")
    return json.loads(content)

# projects = fetch_json(http, '{BRAINMAPS_BASE_URL}/projects')
# volume_list = fetch_json(http, '{BRAINMAPS_BASE_URL}/volumes')

def fetch_subvol_data(http, project, dataset, volume_id, corner, size, scale, change_stack_id=None, subvol_format='raw_snappy'):
    """
    Returns raw subvolume data (not decompressed).
    """
    corner = ','.join(str(x) for x in corner)
    size = ','.join(str(x) for x in size)
    url = ( f'{BRAINMAPS_BASE_URL}/volumes/{project}:{dataset}:{volume_id}'
            f'/binary/subvolume/geometry.corner={corner}/geometry.size={size}/geometry.scale={scale}'
            f'/subvolumeFormat={subvol_format}?alt=media' )

    if change_stack_id:
        url += f'&changeSpec.changeStackId={change_stack_id}'
    
    response, content = http.request(url)
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content}")
    return content

class BrainMapsVolume:
    def __init__(self, project, dataset, volume_id, change_stack_id="", dtype=None, skip_checks=False):
        self.project = project
        self.dataset = dataset
        self.volume_id = volume_id
        self.change_stack_id = change_stack_id
        self.skip_checks = skip_checks
        self._geometries = None
        self._bounding_boxes = None
        self._dtype = dtype

        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if  not credentials_path:
            raise RuntimeError("You must define GOOGLE_APPLICATION_CREDENTIALS in your environment to access BrainMaps volumes.")
        
        scopes = ['https://www.googleapis.com/auth/brainmaps']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scopes)
        self.http = credentials.authorize(Http())
    
        if not self.skip_checks:
            # Verify that the volume exists
            volume_list = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/volumes')
            if f'{project}:{dataset}:{volume_id}' not in volume_list['volumeId']:
                raise RuntimeError(f"BrainMaps volume does not exist on server: {project}:{dataset}:{volume_id}")
    
            if change_stack_id:
                if change_stack_id not in self.get_change_stacks():
                    raise RuntimeError(f"ChangeStackId doesn't exist on the server: '{change_stack_id}'")

        
    @property
    def geometries(self):
        if self._geometries is None:
            self._geometries = self.get_geometry()
            assert int(self._geometries[0]['channelCount']) == 1, \
                "Can't use this class on multi-channel volumes."
        return self._geometries


    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = np.dtype(self.geometries[0]['channelType'].lower())
        return self._dtype


    @property
    def bounding_boxes(self):
        if self._bounding_boxes is None:
            self._bounding_boxes = list(map(self._extract_bounding_box, self.geometries))
        return self._bounding_boxes


    @property
    def bounding_box(self):
        return self.bounding_boxes[0] # Scale 0


    def get_change_stacks(self):
        change_stacks = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/change_stacks')
        return change_stacks['changeStackId']
        

    def get_geometry(self):
        """
        Returns a list of geometries (one per scale):
        
        [{
          'boundingBox': [{
            'corner': {},
            'size': {'x': '37911', 'y': '7731', 'z': '30613'}
          }],
          'channelCount': '1',
          'channelType': 'UINT64',
          'pixelSize': {'x': 8, 'y': 8, 'z': 8},
          'volumeSize': {'x': '37911', 'y': '7731', 'z': '30613'}
        },
        ...]

        (Notice that many of these numbers are strings, for some reason.) 
        """
        geometries = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/volumes/{self.project}:{self.dataset}:{self.volume_id}')['geometry']
        return geometries


    def _extract_bounding_box(self, geometry):
        """
        Return the bounding box [start, stop] in zyx order.
        """
        corner = geometry['boundingBox'][0]['corner']
        size = geometry['boundingBox'][0]['size']

        shape = [int(size[k]) for k in 'zyx']
        if not corner:
            offset = (0,)*len(size)
        else:
            offset = [int(corner[k]) for k in 'zyx']

        box = np.array((offset, offset))
        box[1] += shape
        return box


    def get_subvolume(self, box, scale=0):
        box = np.asarray(box)

        if not self.skip_checks:
            assert (box[0] >= self.bounding_boxes[scale][0]).all() and (box[1] <= self.bounding_boxes[scale][1]).all(), \
                f"Requested box ({box}) extends outside of the volume extents ({self.bounding_box.tolist()})"
        
        corner_zyx = box[0]
        shape_zyx = box[1] - box[0]
        
        corner_xyz = corner_zyx[::-1]
        shape_xyz = shape_zyx[::-1]
        
        snappy_data = fetch_subvol_data( self.http,
                                         self.project,
                                         self.dataset,
                                         self.volume_id,
                                         corner_xyz,
                                         shape_xyz,
                                         scale,
                                         self.change_stack_id,
                                         subvol_format='raw_snappy' )

        volume_buffer = snappy.decompress(snappy_data)
        volume = np.frombuffer(volume_buffer, dtype=self.dtype).reshape(shape_zyx)
        return volume


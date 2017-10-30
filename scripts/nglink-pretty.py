#!/usr/bin/env python
"""
Parse a neuroglancer link (from stdin) into JSON text and print it to the console.
"""
import sys
import json

example_link = \
"""
https://neuroglancer-demo.appspot.com/#!{'layers':{'sec26_image':{'type':'image'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_image'}_'ffn+celis:mask100:threshold0':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask100_0'}_'ffn+celis:mask200:threshold0':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0'_'visible':false}_'ground_truth_bodies_20171017':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_ground_truth_bodies_20171017'_'visible':false}}_'navigation':{'pose':{'position':{'voxelSize':[8_8_8]_'voxelCoordinates':[18955.5_3865.5_15306.5]}}_'zoomFactor':8}}
"""

def replace_commas(d):
    result = {}
    for k,v in d.items():
        new_key = k.replace(',', '_')
        new_val = v
        if isinstance(v, str):
            new_val = v.replace(',', '_')
        
        result[new_key] = new_val
    return result

def pseudo_json_to_data(pseudo_json):
    pseudo_json = pseudo_json.replace('%22', "'")
    pseudo_json = pseudo_json.replace('%7B', '{')
    pseudo_json = pseudo_json.replace('%7D', '}')

    # Make the text valid json by replacing single-quotes
    # with double-quotes and underscores with commas.    
    pseudo_json = pseudo_json.replace("'", '"')
    pseudo_json = pseudo_json.replace("_", ',')
    
    # But underscores within strings should not have been replaced,
    # so change those ones back as we load the json data.
    data = json.loads(pseudo_json, object_hook=replace_commas)
    return data

def main():
    link = sys.stdin.read()
    url_base, pseudo_json = link.split('#!')
    data = pseudo_json_to_data(pseudo_json)
    pretty_text = json.dumps(data, indent=4)

    print("") # Sometimes the terminal prints our ctrl+d control
              # character to the screen as (^D), which messes up the JSON output.
              # Printing a blank line first keeps the json separate.

    print(url_base + '#!    ')
    print(pretty_text)

if __name__ == "__main__":
    main()

import sys
import filecmp

dirpath = sys.argv[1]

path1 = dirpath + "/outputs/blocks"
path2 = dirpath + "/temp_data/blocks"

cmp = filecmp.dircmp(path1, path2)
cmp.report_full_closure()

def find_diff_files(cmp):
    if len(cmp.left_only) > 0 or len(cmp.right_only) > 0 or len(cmp.common_funny) > 0 or len(cmp.diff_files) > 0:
        exit(-1)
    for sub_cmp in cmp.subdirs.values():
        find_diff_files(sub_cmp)

find_diff_files(cmp)


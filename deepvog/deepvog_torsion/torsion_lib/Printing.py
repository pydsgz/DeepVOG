import numpy as np
def print_array_info(img, tag):
    title_string = "=======({})========".format(tag)
    print(title_string)
    print("shape = {}".format(img.shape))
    print("min = {}, max = {}".format(np.min(img), np.max(img)))
    print("="*len(title_string))
import numpy as np

def selections_to_burst(count_min_sketch, threshold, mode=0):
    retVal = [np.inf,np.inf]
    sketch_size = len(count_min_sketch)
    if mode==0:
        i_col = {i for i in range(0,sketch_size) if count_min_sketch[i] > threshold}
        retVal[0]=min(i_col)
        retVal[1]=max(i_col)
    
    return retVal

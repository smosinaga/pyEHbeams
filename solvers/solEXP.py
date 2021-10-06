import numpy as np

def importData(filename):
    "Importing data"
    dataTemp = np.genfromtxt(filename)
    dataTemp[:,0] = dataTemp[:,0]
    dataTemp[:,1] = dataTemp[:,1] * 1 #to add constant conversion if is needed
    
    val = (dataTemp[:,0] , dataTemp[:,1]) #tuple
    return val

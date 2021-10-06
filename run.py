"""
Created on Wed May 27 22:33:09 2020

@author: smosinaga
"""

import time
import sys
from pathlib import Path
sys.path.append(Path(__file__))
import solvers.preProcessor as pre

start = time.time()

modes, sol = pre.read("viga2-TEST")

"""
REMEMBER CHANGE NAME OF THE PLOTS Y auxiliar/plots if you want to save them!!!
"""

print('-------')

# Temporals test
# sol.solveNUMTemp(freq = 110, acc= 2, plot = True)
# sol.solveMMSTemp(freq = 110, acc= 2, plot = True)
# sol.plotCompMMSandNUM()

# sol.solveMMSTemp(freq = 150, acc= 1, plot = True)

end = time.time()
print('Finished in '+str(round(end-start,5))+'s')

del(end); del(start)



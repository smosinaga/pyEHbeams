from models import modes
from models import system
from solvers import solve as ss

import importlib, json, pathlib

def read(caseName):
    # Open the input file
    casePath = pathlib.Path(__file__).parent.absolute()/".."/"input"
    caseName += ".json"
    with open(casePath/caseName, 'r') as f:
        caseDict = json.load(f) # Load the input file

    # Solving modes
    modesSol = getattr(modes, caseDict['device'])(caseDict)

    # Matrix assembly and ODE pars
    systemSol = getattr(system, caseDict['formulation'])(caseDict,modesSol)
    
    # Solving for for MMS, numerical integration and exporting experimental data
    sol = ss.solution(caseDict,systemSol)
    
    return modesSol, sol
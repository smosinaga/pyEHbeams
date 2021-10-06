
 ==============================================================================
 "device": Needed to solve mode shapes
 ==============================================================================

"AxialLoadedBeam": Axial loaded fixed-fixed beam. Reduced model by considerate O(u) = O(w^2)
"CantileverBeam2": A two section cantilever beam where the piezo is in the first section.
"CantileverBeam3": A three section cantilever beam where the piezo is in the second section.

 ==============================================================================
 "formulation": 
 ==============================================================================
"NonLinear": Just supported for AxialLoadedBeam.
"Linear": Supported for all models

 ==============================================================================
 "expSample": 
 ==============================================================================
List. Prefix of .txt files in experimental folder
 
 ==============================================================================
  "setup":
 ==============================================================================

"solve_disp": "y" or "n"
"solve_volt": "y" or "n"

Indicates if solutions for displacement or voltage are expected

"solve_MMS": "y" or "n"
"solve_Num": "y" or "n"
"solve_Exp": ""y" or "n"
 
Indictates if solutions for Multiple Scales Method, numerical integration,
experimental results must be obteined

 ==============================================================================
 "acelValues":
 ==============================================================================
 List. Acceleration values in g-unit to solve the system

 ==============================================================================
 "solRanges"
 ==============================================================================

"fiMSM": Initial frequency for MMS method
"ffMSM": Final frequency for MMS method,
"minMSM": Minumun excpected displacement for MSM method
"maxMSM": Maximum expected displacement for MSM method
"nMSM": Number of values to evaluate MSM solution

Solutions are first searched between minMSM and maxMSM method and then cutted 
beetween fiMSM and ffMSM

"fi1": Initial frequency for Numerical integration (sweep 1)
"ff1": Final frequency for Numerical integration (sweep 1)
"n1": Numer of frequencies evaluation (sweep 1)

"fi2": Initial frequency for Numerical integration (sweep 2)
"ff2": Final frequency for Numerical integration (sweep 2)
"n2": Numer of frequencies evaluation (sweep 2)

"fi3": Initial frequency for Numerical integration (sweep 3)
"ff3": Final frequency for Numerical integration (sweep 3)
"n3": Number of frequencies evaluation (sweep 3)

 ==============================================================================
 "modesConf"
 ==============================================================================

"wi": Minumun circular frequency value for natural frequencies search,
"wf": Maximum circular frequency value for natural frequencies search,
"sampleNum": Number of evaluation of the characteristic equation, 
"modeResolution": mode resolution for each section (number of discrate values),

"plot_MatFreq": "y" or "n"
"plot_Modes": "y" or "n"

Indicates if plots of Determinat of the equation matrix and mode shapes are 
expected or not.

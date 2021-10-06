import numpy as np
from scipy import integrate

beta = 0.65

# =============================================================================
# Main class for ODE pars
# =============================================================================

class DiscreteSystem(object):
    def __repr__(self): return 'Discrete_element'
    def __init__(self, input, modes):

        # Exporting data from solved modes
        self.xcor = modes.xcor
        self.geoDef = modes.geoDef
        self.NaturalFreq = modes.NaturalFreq
        self.SolvedModes = modes.SolvedModes #[mode][section]
        self.DerMode = modes.DerMode #[mode][der][section]
        self.pars = modes.pars
        
        # Number of modes and sections
        self.nModes = len(self.NaturalFreq)
        self.nSec = len(self.xcor)
        
        # Building rho, EA, EI and L lists
        self.rhoA = []
        for flag in range(self.nSec):
            rhoName = 'rhoA'+str(flag + 1)
            self.rhoA.append( self.geoDef [rhoName])
        
        self.EA = []
        for flag in range(self.nSec):
            EAName = 'EA'+str(flag + 1)
            self.EA.append( self.geoDef [EAName])
            
        self.EI = []
        for flag in range(self.nSec):
            EIName = 'EI'+str(flag + 1)
            self.EI.append( self.geoDef [EIName])
            
        self.L = []
        for flag in range(self.nSec):
            LName = 'L'+str(flag + 1)
            self.L.append( self.pars [LName])
        
        #Finding piezo position (where mass per unit is bigger)
        self.piezoPos = self.rhoA.index(max(self.rhoA))
        
        # Init empty matrix (Linear and nonlinear terms)
        self.emptyMatrix()
        
        # Matrix Assembly (Linear terms)
        self.assyForceVector()
        self.assyMassMatrix()
        self.assyStiffMatrix()
        self.assyLinDampMatrix()
        self.assyLinCouplingVector()
        self.assyScalarValues()
        
        # Mode evaluated at specific position
        self.assyModeEval() 
      
    def emptyMatrix(self):     
        # Linear
        self.FF = np.zeros( self.nModes )
        self.MM = np.zeros( [self.nModes,self.nModes] )
        self.KK = np.zeros( [self.nModes,self.nModes] )
        self.CC = np.zeros( [self.nModes,self.nModes] )
        self.theta = np.zeros( [self.nModes] ) 
        self.psi = np.zeros( [self.nModes] )
        
        # NonLinear parameters
        self.MMM = np.zeros( [self.nModes, self.nModes, self.nModes] )
        self.CCC = np.zeros( [self.nModes,self.nModes,self.nModes] )   
        self.Kg = np.zeros( [self.nModes, self.nModes, self.nModes, self.nModes] ) #Nonlinear geometric stiffness (O(3))
        self.thetaG = np.zeros( [self.nModes, self.nModes] )
        self.psiG = np.zeros( [self.nModes, self.nModes] )
        self.thetaC = np.zeros( [self.nModes, self.nModes] )
        self.psiC = np.zeros( [self.nModes, self.nModes] )
        self.Kc = np.zeros( [self.nModes, self.nModes, self.nModes] )
        
        #Nonlinear parameters from postbuckling conf
        self.Kgq = np.zeros( [self.nModes, self.nModes, self.nModes] )
        self.Kcq = np.zeros( [self.nModes, self.nModes] )
        self.thetaGq = np.zeros( [self.nModes] )
        self.thetaCq = np.zeros( [self.nModes] )
        self.psiGq = np.zeros( [self.nModes] )
        self.psiCq = np.zeros( [self.nModes] )
        
        #Evaluated modes at specific position
        self.modeEval = np.zeros( self.nModes )
        
    def assyForceVector(self):
        g_const = 9.81    
         
        for i in range(self.nModes):
            el = 0
            for sec in range(self.nSec):
                el += g_const * self.rhoA[sec] * integrate.trapz(self.SolvedModes[i][sec], self.xcor[sec])   
            self.FF[i] = el
            
    def assyMassMatrix(self):
        for i in range(self.nModes):
            for j in range(self.nModes):
                el = 0
                for sec in range(self.nSec):
                    if i is j:
                        el += self.rhoA[sec] * integrate.trapz(self.SolvedModes[i][sec] * self.SolvedModes[j][sec] , self.xcor[sec])   
                    else:
                        el = 0 #forced to be diagonal matrix
                self.MM[i,j] = el   
    
    def assyStiffMatrix(self):
        for i in range(self.nModes):
            self.KK[i,i] = self.NaturalFreq[i]**2
        
    def assyLinDampMatrix(self):
        damp = self.pars['c1']
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                el = 0
                for sec in range(self.nSec):
                    if i is j:
                        el += damp * integrate.trapz(self.SolvedModes[i][sec] * self.SolvedModes[j][sec] , self.xcor[sec])   
                    else:
                        el = 0 #forced to be diagonal matrix
                self.CC[i,j] = el
        
    def assyLinCouplingVector(self):
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            el =  self.geoDef['theta1'] * ( self.DerMode[i][0][sec][1] - self.DerMode[i][0][sec][-1]  )
            self.theta[i] = el
            self.psi[i] = - beta*el
    
    def assyScalarValues(self):
        self.Rl = self.pars['Rl']
        self.Cp = self.pars['Cp']
        
    def assyModeEval(self):  
        vib_pos = self.pars['vib_pos']
        
        for i in range(self.nModes):
            x_len = np.array(0) #Not the best way to do it...i know!
            y_len = np.array(0)
            for sec in range(self.nSec):
                x_len = np.append(x_len, [ self.xcor[sec] ] )
                y_len = np.append(y_len, [ self.SolvedModes[i][sec] ] )
            
            pos = np.where (x_len>=vib_pos)[0][0] #First element 
            el = y_len[pos]
            
            self.modeEval[i] = el
    
  
# =============================================================================
# Linear formulation Class
# =============================================================================

class Linear(DiscreteSystem):
        def __init__(self,input,modes):
            super().__init__(input,modes) #Inheritance

# =============================================================================
# Nonlinear formulation class - Prebuckling state
# Axial BC for the reduction: clamped-axial force + SPRING
# =============================================================================

class NonLinearAxialSpringPre(DiscreteSystem):
    def __init__(self,input,modes):
        super().__init__(input,modes) #Inheritance

        self.assyNonLinInertia()        
        self.assyNonLinDampMatrix()
        self.assyNonLinGeoMatrix()
        self.assyNonLinGeoCoupling()
        self.assyNonLinConCoupling()
        self.assyNonLinConStiffness()
        
    def assyNonLinInertia(self):
        chi = self.geoDef["chi"]
        
        for i in range (self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for sec in range(self.nSec):
                        el = chi * integrate.trapz (self.SolvedModes[i][sec] * self.DerMode[j][0][sec]*self.DerMode[k][0][sec])
                        
                        self.MMM[i,j,k] = el
            
    def assyNonLinDampMatrix(self):
        damp = self.pars['c2']
        # sec = self.piezoPos 
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for sec in range(self.nModes):
                        el = damp * integrate.trapz(self.SolvedModes[i][sec] * self.SolvedModes[j][sec] * self.SolvedModes[k][sec]*np.sign( self.SolvedModes[j][sec]) , self.xcor[sec])   
                    
                    self.CCC[i,j,k] = el
    
    def assyNonLinGeoMatrix(self):
        kr = self.pars["k"]
        alpha = self.geoDef["alpha"]

        
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for l in range(self.nModes):
                        el = 0
                        integral = 0
                        
                        for sec in range(self.nSec):
         
                            integral += integrate.trapz( self.DerMode[k][0][sec]*self.DerMode[l][0][sec], self.xcor[sec] )
                        
                        for sec in range(self.nSec):
                            # nayfeh
                            # el += 0.5*self.EA[sec] * ( self.L[0] +  self.L[1] + self.L[2] )**-1 * integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[k][1][sec], self.xcor[sec] )
                            
                            el += 0.5*kr*alpha*integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[k][1][sec], self.xcor[sec] )
                        
                        self.Kg[i,j,k,l] = -el*integral
    
    def assyNonLinGeoCoupling(self):   
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                el1 = self.geoDef['theta0'] * integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][1][sec] , self.xcor[sec]) 
                el2 = self.geoDef['theta0'] * self.geoDef['k'] *self.L[sec]* self.geoDef['alpha'] * (self.EA[sec])**-1 * \
                            integrate.trapz( self.DerMode[i][0][sec] * self.DerMode[j][0][sec] , self.xcor[sec]) 
                
                self.thetaG[i,j] = el1
                self.psiG[i,j] = beta*el2
            
    def assyNonLinConCoupling(self):
        sec = self.piezoPos 
       
        for i in range (self.nModes):
            for j in range (self.nModes):
                    el1 =   self.geoDef['psi2'] * integrate.trapz ( self.SolvedModes[i][sec] * self.DerMode[j][3][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec] )              
                    el2 =   self.geoDef['psi2'] * integrate.trapz ( self.DerMode[i][1][sec]  * self.DerMode[j][1][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec] )
                    
                    self.thetaC[i,j] = el1
                    self.psiC[i,j] = - beta*el2
            
    def assyNonLinConStiffness(self):  
        sec = self.piezoPos 
        
        for i in range (self.nModes):
            for j in range (self.nModes):
                for k in range (self.nModes):
                    el = 2 * self.geoDef['EJn']* integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][2][sec] * self.DerMode[k][2][sec] * np.sign(self.DerMode[j][1][sec]) + 
                                                                  self.SolvedModes[i][sec] * self.DerMode[j][1][sec] * self.DerMode[k][3][sec] * np.sign(self.DerMode[j][1][sec]),\
                                   self.xcor[sec] )  

                    self.Kc[i,j,k] = el

# =============================================================================
# Nonlinear formulation class - Postbuckling state
# Axial BC for the reduction: clamped-axial force + SPRING
# =============================================================================

class NonLinearAxialSpringPost(NonLinearAxialSpringPre):
    def __init__(self,input,modes):
        #buckled modes must be imported in this class
        self.SolvedBuckModes = modes.SolvedBuckModes #[mode][section]
        self.DerBuckMode = modes.DerBuckMode #[mode][der][section]
        
        super().__init__(input,modes) #Inheritance
        
        self.assyNonLinGeoMatrix2()
        self.assyNonLinGeoCoupling2()
        self.assyNonLinConCoupling2()
        self.assyNonLinConStiffness2()
            

    def assyNonLinGeoMatrix2(self):
        kr = self.pars["k"]
        alpha = self.geoDef["alpha"]
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):                 
                    el1 = 0; el2 = 0; integral1 = 0; integral2 = 0
                    
                    for sec in range(self.nSec):
                        integral1 += 0.5 * integrate.trapz( self.DerMode[j][0][sec]     * self.DerMode[k][0][sec], self.xcor[sec] )
                        integral2 +=       integrate.trapz( self.DerBuckMode[0][0][sec] * self.DerMode[k][0][sec], self.xcor[sec] )
 
                    for sec in range(self.nSec):
                        el1 += kr*alpha*integrate.trapz( self.SolvedModes[i][sec] * self.DerBuckMode[0][1][sec], self.xcor[sec] )
                        el2 += kr*alpha*integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][1][sec]    , self.xcor[sec] )
                    
                    self.Kgq[i,j,k] = - el1*integral1 - el2*integral2
    
    def assyNonLinGeoCoupling2(self):   
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            el1 = self.geoDef['theta0'] * integrate.trapz( self.SolvedModes[i][sec] * self.DerBuckMode[0][1][sec] , self.xcor[sec]) 
            el2 = self.geoDef['theta0'] * 2* self.geoDef['k'] *self.L[sec]* self.geoDef['alpha'] * (self.EA[sec])**-1 * \
                                         integrate.trapz( self.DerMode[i][0][sec] * self.DerBuckMode[0][0][sec] , self.xcor[sec]) 
            
            self.thetaGq[i] = el1
            self.psiGq[i] = beta*el2
            
    def assyNonLinConCoupling2(self):
        sec = self.piezoPos 
       
        for i in range (self.nModes):
            el1 =   self.geoDef['psi2'] * integrate.trapz ( self.SolvedModes[i][sec] * self.DerBuckMode[0][3][sec] * np.sign(self.DerMode[i][1][sec]) , self.xcor[sec] )              
            el2 =   self.geoDef['psi2'] * integrate.trapz ( self.DerMode[i][1][sec]  * self.DerBuckMode[0][1][sec] * np.sign(self.DerMode[i][1][sec]) , self.xcor[sec] )
            
            self.thetaCq[i] = el1
            self.psiCq[i] =  - beta*el2

    def assyNonLinConStiffness2(self):
        sec = self.piezoPos 
        
        for i in range (self.nModes):
            for j in range (self.nModes):
                el =     self.geoDef['EJn']* integrate.trapz(4 * self.SolvedModes[i][sec] * self.DerBuckMode[0][2][sec] * self.DerMode[j][2][sec] * np.sign(self.DerMode[j][1][sec]) + 
                                                             2 * self.SolvedModes[i][sec] * self.DerBuckMode[0][3][sec] * self.DerMode[j][1][sec] * np.sign(self.DerMode[j][1][sec]) + 
                                                             2 * self.SolvedModes[i][sec] * self.DerBuckMode[0][1][sec] * self.DerMode[j][3][sec] * np.sign(self.DerMode[j][1][sec]),\
                               self.xcor[sec] )  
        
                self.Kcq[i,j] = el

# =============================================================================
# Nonlinear formulation class - Prebuckling state
# Axial BC for the reduction: clamped-axial force (similar Masana et al.)
# =============================================================================

class NonLinearAxialForce(DiscreteSystem):
    def __init__(self,input,modes):
        super().__init__(input,modes) #Inheritance

        self.assyNonLinInertia()        
        self.assyNonLinDampMatrix()
        self.assyNonLinGeoMatrix()
        self.assyNonLinGeoCoupling()
        self.assyNonLinConCoupling()
        self.assyNonLinConStiffness()
        
    def assyNonLinInertia(self):
        chi = self.geoDef["chi"]
        
        for i in range (self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for sec in range(self.nSec):
                        el = chi * integrate.trapz (self.SolvedModes[i][sec] * self.DerMode[j][0][sec]*self.DerMode[k][0][sec])
                        
                        self.MMM[i,j,k] = el
        
    def assyNonLinDampMatrix(self):
        damp = self.pars['c2']
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    el = damp * integrate.trapz(self.SolvedModes[i][sec] * self.SolvedModes[j][sec] * self.SolvedModes[k][sec]*np.sign( self.SolvedModes[j][sec]) , self.xcor[sec])   
                    
                    self.CCC[i,j,k] = el
    
    def assyNonLinGeoMatrix(self):
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for l in range(self.nModes):
                        el = 0
                        
                        for sec in range(self.nSec):  
                            el +=  -(self.EI[sec]) *\
                            integrate.trapz( 0.5 * self.SolvedModes[i][sec] * self.DerMode[j][1][sec] * self.DerMode[k][1][sec] * self.DerMode[l][1][sec] + \
                                                   self.SolvedModes[i][sec] * self.DerMode[j][0][sec] * self.DerMode[k][1][sec] * self.DerMode[l][2][sec]
                                            , self.xcor[sec] )  
                                
                        self.Kg[i,j,k,l] = el
    
    def assyNonLinGeoCoupling(self):   
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                el1 = self.geoDef['theta0'] * integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][1][sec] , self.xcor[sec]) 
                el2 = self.geoDef['theta2'] * integrate.trapz( self.DerMode[i][1][sec]  * self.DerMode[j][1][sec] , self.xcor[sec])
                
                self.thetaG[i,j] = el1
                self.psiG[i,j] = beta*el2
            
    def assyNonLinConCoupling(self):
        sec = self.piezoPos 
       
        for i in range (self.nModes):
            for j in range (self.nModes):
                el1 =   self.geoDef['psi2'] * integrate.trapz ( self.SolvedModes[i][sec] * self.DerMode[j][3][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec] )    
                el2 = - self.geoDef['psi2'] * integrate.trapz ( self.DerMode[i][1][sec]  * self.DerMode[j][1][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec] )
                
                self.thetaC[i,j] = el1
                self.psiC[i,j] = beta*el2
            
    def assyNonLinConStiffness(self):  
        sec = self.piezoPos 
        
        for i in range (self.nModes):
            for j in range (self.nModes):
                for k in range (self.nModes):
                    el = 2 * self.geoDef['EJn']* integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][2][sec] * self.DerMode[k][2][sec] * np.sign(self.DerMode[j][1][sec]) + 
                                                                  self.SolvedModes[i][sec] * self.DerMode[j][1][sec] * self.DerMode[k][3][sec] * np.sign(self.DerMode[j][1][sec]),\
                                   self.xcor[sec] )  
            
                    self.Kc[i,j,k] = el

# =============================================================================
# Nonlinear formulation class
# Parameters according Gatti's Thesis 
# Inextensible beam assumption
# =============================================================================        

class NonLinearGatti(DiscreteSystem):
    
    def __init__(self,input,modes):
        super().__init__(input,modes) #Inheritance

        self.assyNonLinDampMatrix()
        self.assyNonLinGeoMatrix()
        self.assyNonLinGeoCoupling()
        self.assyNonLinConCoupling()
        self.assyNonLinConStiffness()
        self.assyNonlinIntertia()
            
    def assyNonLinDampMatrix(self):
        damp = self.pars['c2']
        sec = self.piezoPos 
        
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    el = damp * integrate.trapz(self.SolvedModes[i][sec] * self.SolvedModes[j][sec] * self.SolvedModes[k][sec]*np.sign( self.SolvedModes[j][sec]) , self.xcor[sec])   
                    
                    self.CCC[i,j,k] = el
    
    def assyNonLinGeoMatrix(self):
        for i in range(self.nModes):
            for j in range(self.nModes):
                for k in range(self.nModes):
                    for l in range(self.nModes):
                        el = 0
                        for sec in range(self.nSec):

                            el +=  (self.EI[sec]) *\
                                    integrate.trapz(     self.SolvedModes[i][sec] * self.DerMode[j][1][sec] * self.DerMode[k][1][sec] * self.DerMode[l][1][sec] + \
                                                     4 * self.SolvedModes[i][sec] * self.DerMode[j][0][sec] * self.DerMode[k][1][sec] * self.DerMode[l][2][sec] + \
                                                         self.SolvedModes[i][sec] * self.DerMode[j][0][sec] * self.DerMode[k][0][sec] * self.DerMode[l][3][sec]
                                            , self.xcor[sec] ) 
                        
                        self.Kg[i,j,k,l] = el
    
    def assyNonLinGeoCoupling(self):   
        sec = self.piezoPos 

        for i in range(self.nModes):
            for j in range(self.nModes):
                el1 = 0 #not considerated...in my formulation this terms is O(v*q) and for Claudio (v*q**2)
                el2 = 0 #same as el1
                
                self.thetaG[i,j] = el1
                self.psiG[i,j] = beta*el2
            
    def assyNonLinConCoupling(self):
        sec = self.piezoPos 
       
        for i in range (self.nModes):
            for j in range (self.nModes):
                el1 = -self.geoDef['psi2'] * integrate.trapz ( self.SolvedModes[i][sec] * self.DerMode[j][3][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec] )
                el2 =  self.geoDef['psi2'] * integrate.trapz ( self.DerMode[i][1][sec]  * self.DerMode[j][1][sec] * np.sign(self.DerMode[j][1][sec]) , self.xcor[sec]  )

                self.thetaC[i,j] = el1
                self.psiC[i,j] = beta*el2
            
    def assyNonLinConStiffness(self):  
        sec = self.piezoPos 
        
        for i in range (self.nModes):
            for j in range (self.nModes):
                for k in range (self.nModes):
                    el = 2 * self.geoDef['EJn']* integrate.trapz( self.SolvedModes[i][sec] * self.DerMode[j][2][sec] * self.DerMode[k][2][sec] * np.sign(self.DerMode[j][1][sec]) + 
                                                                  self.SolvedModes[i][sec] * self.DerMode[j][1][sec] * self.DerMode[k][3][sec] * np.sign(self.DerMode[j][1][sec]),\
                                   self.xcor[sec] )  
            
                    self.Kc[i,j,k] = el
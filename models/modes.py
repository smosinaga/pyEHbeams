import sys
import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import integrate
from auxiliar import usedFun as nf
from auxiliar import plots as pp

# =============================================================================
# Parent class
# =============================================================================

class Modes(object):
    def __repr__(self): return 'Mode_element'
    def __init__(self, input):
        self.pars = input["pars"]
        self.modesConf = input["modesConf"]
        
    def DetMatFreq(self,omega):
        det = linalg.det( self.MatrixFreq(omega) )
        return det

    def DetMatBuck(self,Pval):
        det = linalg.det( self.MatrixBuck(Pval) )
        return det
    
    def FindNaturalFreq(self):
        wi = self.modesConf['wi']; wf = self.modesConf['wf']; sample = self.modesConf['sampleNum']
         
        wValues =  np.linspace(wi,wf,sample)
        MatFreqVal = [self.DetMatFreq(x) for x in wValues]
        
        if self.modesConf['plot_MatFreq'] == 'y':
            pp.plotFreqEquation(wValues,MatFreqVal)
        elif self.modesConf['plot_MatFreq'] == 'n':
            pass
        else:
            print("plot_MatFreq must be 'y' or 'n'")
            sys.exit()
        
        initGuess = [] #Initial guess to be used in Newton-Rapshon method (change of sign)
        for idx in range(len(wValues)-1):
            if np.sign(MatFreqVal[idx]) != np.sign(MatFreqVal[idx+1]):
                initGuess.append(wValues[idx])
        
        wn = [] #Natural frequencies by Newton-Rapshon method
        for idx in range(len(initGuess)):
            wn.append(optimize.newton(self.DetMatFreq, initGuess[idx],maxiter = 500, tol=1e-12))
        
        wn.sort() #Sortering natural frequencies 
        self.NaturalFreq = wn
        print('Natural frequencies finded:')
        for j in self.NaturalFreq:
            print(round(j*(2*np.pi)**-1,2))
            
    def FindCriticalLoad(self):
        Pi = self.bucklingConf['Pi']; Pf = self.bucklingConf['Pf']; sample = self.bucklingConf['sampleNum']
         
        PValues =  np.linspace(Pi,Pf,sample)
        MatBuckVal = [self.DetMatBuck(x) for x in PValues]
        
        if self.bucklingConf['plot_buckFreq'] == 'y':
            pp.plotBuckEquation(PValues,MatBuckVal)
        elif self.bucklingConf['plot_buckFreq'] == 'n':
            pass
        else:
            print("plot_buckFreq must be 'y' or 'n'")
            sys.exit()
        
        initGuess = [] #Initial guess to be used in Newton-Rapshon method (change of sign)
        for idx in range(len(PValues)-1):
            if np.sign(MatBuckVal[idx]) != np.sign(MatBuckVal[idx+1]):
                initGuess.append(PValues[idx])
        
        Ptest = [] #Natural frequencies by Newton-Rapshon method
        for idx in range(len(initGuess)):
            Ptest.append(optimize.newton(self.DetMatBuck, initGuess[idx],maxiter = 500, tol=1e-12))
        
        Ptest.sort() #Sortering natural frequencies 
        self.Pcrit = Ptest
        print('Critical values for P founded:')
        for j in self.Pcrit:
            print(round(j,2))
            
    def SolveDerivates(self, n = 4):
        # n- is the max order of the derivate
        numSections = len(self.xcor)
        
        self.DerMode = []    
        for mode in self.SolvedModes:
            aux2 = []
            for order in range(1, n+1 ):
                aux1 = []
                for sec in range(numSections):    
                    der = nf.NumDer(self.xcor[sec],mode[sec],order)
                    aux1.append(der)
                aux2.append(aux1)
            self.DerMode.append(aux2)

    def SolveDerivativesBuck(self, n = 4):
        # n- is the max order of the derivate
        numSections = len(self.xcor)
        
        self.DerBuckMode = []    
        for mode in self.SolvedBuckModes:
            aux2 = []
            for order in range(1, n+1 ):
                aux1 = []
                for sec in range(numSections):    
                    der = nf.NumDer(self.xcor[sec],mode[sec],order)
                    aux1.append(der)
                aux2.append(aux1)
            self.DerBuckMode.append(aux2)

    def PlotModes(self):
        if self.modesConf['plot_Modes'] == 'y':
            pp.plotModes(self)
        elif self.modesConf['plot_Modes'] == 'n':
            pass
        else:
            print("plot_Modes must be 'y' or 'n'")
            sys.exit()

    def PlotBuckModes(self):
        if self.bucklingConf['plot_buckModes'] == 'y':
            pp.plotBuckModes(self)
        elif self.bucklingConf['plot_buckModes'] == 'n':
            pass
        else:
            print("plot_buckModes must be 'y' or 'n'")
            sys.exit()
    
    def MatrixFreq(self):
        #must be implemented in the subclass
        raise NotImplementedError

    def SolveModes(self):
        #must be implemented in the subclass
        raise NotImplementedError
        
# =============================================================================
# Axially loaded prebuckled beam - applied axial force
# =============================================================================

class AxialPreBuckledBeam3(Modes):
    def __init__(self,input):
        super().__init__(input) #Inheritance
        
        hs = self.pars['hs']; bs = self.pars['bs']; Es = self.pars['Es']
        hp = self.pars['hp']; bp = self.pars['bp']; Ep = self.pars['Ep']
        L1 = self.pars['L1']; L2 = self.pars['L2']; L3 = self.pars['L3']
        
        rhos = self.pars['rhos']; rhop = self.pars['rhop']; P = self.pars['P']
        
        e31 = self.pars['d31']*self.pars['Ep']; e311 = self.pars['e311']
        c111p = self.pars['c111p'] 
  
        self.geoDef = {
            'rhoA1': bs*hs*rhos,
            'rhoA2': ( bp*hp*rhop + bs*hs*rhos ),
            'rhoA3': bs*hs*rhos,
            'EI1': 1/12*bs*Es*(hs)**(3),
            'EI2': 1/12*((bp*Ep*hp + bs*Es*hs ))**(-1) * ((bp)**(2)*(Ep)**(2)*(hp)**(4) + ((bs)**(2)*(Es)**(2)*(hs)**(4) + 2*bp*bs*Ep*Es*hp*hs*(2*(hp)**(2) + (3*hp*hs + 2*(hs)**(2))))),
            'EI3': 1/12*bs*Es*(hs)**(3),
            
            'EA1': bs*hs*Es,
            'EA2': bs*hs*Es + bp*hp*Ep,
            'EA3': bs*hs*Es,
        
            'L1': L1, 'L2': L2, 'L3': L3,
            'P':P,
            
            'EAn': bp * c111p * hp,
            'EBn': 1/2 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'EIn': 1/12 * bp * c111p * hp * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'EJn': 1/8 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -3 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 2 * ( hp )**( 2 ) + ( 2 * hp * hs + ( hs )**( 2 ) ) ) ) ),
            
            'theta0': bp * e31,
            'theta1': 1/2 * bp * bs * Es * e31 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'theta2': 1/12 * bp * e31 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'psi0': bp * e311,
            'psi1': 1/2 * bp * bs * Es * e311 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'psi2': 1/12 * bp * e311 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            
            'chi': 1/2 * bp * bs * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ) * ( -1 * Es * rhop + Ep * rhos )
            }
        
        self.SolveModes()
        self.SolveDerivates()
        self.PlotModes()
        
    def MatrixFreq(self,omega):
        geoDef = self.geoDef 
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] )**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] )**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] )**(1/2)
    
        beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
        beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
        beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
    
        s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )

        MatFreq = np.zeros([12,12]) #Initial empty matrix

        #Coefficients
        MatFreq[0,0] = 1
        MatFreq[0,1] = 0
        MatFreq[0,2] = 1
        MatFreq[0,3] = 0
        MatFreq[0,4] = 0
        MatFreq[0,5] = 0
        MatFreq[0,6] = 0
        MatFreq[0,7] = 0
        MatFreq[0,8] = 0
        MatFreq[0,9] = 0
        MatFreq[0,10] = 0
        MatFreq[0,11] = 0
    
        MatFreq[1,0] = 0
        MatFreq[1,1] = s11
        MatFreq[1,2] = 0
        MatFreq[1,3] = s12
        MatFreq[1,4] = 0
        MatFreq[1,5] = 0
        MatFreq[1,6] = 0
        MatFreq[1,7] = 0
        MatFreq[1,8] = 0
        MatFreq[1,9] = 0
        MatFreq[1,10] = 0
        MatFreq[1,11] = 0
    
        MatFreq[2,0] = np.cosh( geoDef['L1'] * s11 )
        MatFreq[2,1] = np.sinh( geoDef['L1'] * s11 )
        MatFreq[2,2] = np.cos( geoDef['L1'] * s12 )
        MatFreq[2,3] = np.sin( geoDef['L1'] * s12 )
        MatFreq[2,4] = -1 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[2,5] = -1 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[2,6] = -1 * np.cos( geoDef['L1'] * s22 )
        MatFreq[2,7] = -1 * np.sin( geoDef['L1'] * s22 )
        MatFreq[2,8] = 0
        MatFreq[2,9] = 0
        MatFreq[2,10] = 0
        MatFreq[2,11] = 0
    
        MatFreq[3,0] = s11 * np.sinh( geoDef['L1'] * s11 )
        MatFreq[3,1] = s11 * np.cosh( geoDef['L1'] * s11 )
        MatFreq[3,2] = -1 * s12 * np.sin( geoDef['L1'] * s12 )
        MatFreq[3,3] = s12 * np.cos( geoDef['L1'] * s12 )
        MatFreq[3,4] = -1 * s21 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[3,5] = -1 * s21 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[3,6] = s22 * np.sin( geoDef['L1'] * s22 )
        MatFreq[3,7] = -1 * s22 * np.cos( geoDef['L1'] * s22 )
        MatFreq[3,8] = 0
        MatFreq[3,9] = 0
        MatFreq[3,10] = 0
        MatFreq[3,11] = 0
    
        MatFreq[4,0] = geoDef['EI1'] * ( s11 )**( 2 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[4,1] = geoDef['EI1'] * ( s11 )**( 2 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[4,2] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[4,3] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[4,4] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[4,5] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[4,6] = geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[4,7] = geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[4,8] = 0
        MatFreq[4,9] = 0
        MatFreq[4,10] = 0
        MatFreq[4,11] = 0
    
        MatFreq[5,0] = geoDef['EI1'] * ( s11 )**( 3 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[5,1] = geoDef['EI1'] * ( s11 )**( 3 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[5,2] = geoDef['EI1'] * ( s12 )**( 3 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[5,3] = -1 * geoDef['EI1'] * ( s12 )**( 3 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[5,4] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[5,5] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[5,6] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[5,7] = geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[5,8] = 0
        MatFreq[5,9] = 0
        MatFreq[5,10] = 0
        MatFreq[5,11] = 0
    
        MatFreq[6,0] = 0
        MatFreq[6,1] = 0
        MatFreq[6,2] = 0
        MatFreq[6,3] = 0
        MatFreq[6,4] = np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,5] = np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,6] = np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,7] = np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,8] = -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,9] = -1 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,10] = -1 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[6,11] = -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[7,0] = 0
        MatFreq[7,1] = 0
        MatFreq[7,2] = 0
        MatFreq[7,3] = 0
        MatFreq[7,4] = s21 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,5] = s21 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,6] = -1 * s22 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,7] = s22 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,8] = -1 * s31 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,9] = -1 * s31 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,10] = s32 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[7,11] = -1 * s32 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[8,0] = 0
        MatFreq[8,1] = 0
        MatFreq[8,2] = 0
        MatFreq[8,3] = 0
        MatFreq[8,4] = geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,5] = geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,6] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,7] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,8] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,9] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,10] = geoDef['EI3'] * ( s32 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[8,11] = geoDef['EI3'] * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[9,0] = 0
        MatFreq[9,1] = 0
        MatFreq[9,2] = 0
        MatFreq[9,3] = 0
        MatFreq[9,4] = geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,5] = geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,6] = geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,7] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,8] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,9] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,10] = -1 * geoDef['EI3'] * ( s32 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[9,11] = geoDef['EI3'] * ( s32 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[10,0] = 0
        MatFreq[10,1] = 0
        MatFreq[10,2] = 0
        MatFreq[10,3] = 0
        MatFreq[10,4] = 0
        MatFreq[10,5] = 0
        MatFreq[10,6] = 0
        MatFreq[10,7] = 0
        MatFreq[10,8] = np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,9] = np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,10] = np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[10,11] = np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
    
        MatFreq[11,0] = 0
        MatFreq[11,1] = s11
        MatFreq[11,2] = 0
        MatFreq[11,3] = s12
        MatFreq[11,4] = 0
        MatFreq[11,5] = 0
        MatFreq[11,6] = 0
        MatFreq[11,7] = 0
        MatFreq[11,8] = s31 * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,9] = s31 * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,10] = -1 * s32 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[11,11] = s32 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        
        return MatFreq
        
    def SolveModes(self):
        
        self.FindNaturalFreq()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution']
             
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation of the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)
        x3 = np.linspace(geoDef['L1'] + geoDef['L2'] ,geoDef['L1'] + geoDef['L2'] + geoDef['L3'] , num = modeRes)
        
        self.xcor = [x1,x2,x3]
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] )**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] )**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] )**(1/2)
        
        SolvedModes = []
        
        for idx,omega in enumerate(self.NaturalFreq):          
            beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
            beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
            beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
        
            s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )

            C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12 = nf.SolveLinSys( self.MatrixFreq(omega) )
            
            shapeLen1 = lambda x: ( C3 * np.cos( s12 * x ) + ( C1 * np.cosh( s11 * x ) + ( C4 * np.sin( s12 * x ) + C2 * np.sinh( s11 * x ) ) ) )
            shapeLen2 = lambda x: ( C7 * np.cos( s22 * x ) + ( C5 * np.cosh( s21 * x ) + ( C8 * np.sin( s22 * x ) + C6 * np.sinh( s21 * x ) ) ) )
            shapeLen3 = lambda x: ( C11 * np.cos( s32 * x ) + ( C9 * np.cosh( s31 * x ) + ( C12 * np.sin( s32 * x ) + C10 * np.sinh( s31 * x ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1)
            evalShapeLen2 = shapeLen2(x2)
            evalShapeLen3 = shapeLen3(x3)
            
            conOrt = integrate.trapz(geoDef['rhoA1']*evalShapeLen1**2,x1) + \
                     integrate.trapz(geoDef['rhoA2']*evalShapeLen2**2,x2) + \
                      integrate.trapz(geoDef['rhoA3']*evalShapeLen3**2,x3)
            
            mod1 = evalShapeLen1 * np.sqrt(1/conOrt)
            mod2 = evalShapeLen2 * np.sqrt(1/conOrt)
            mod3 = evalShapeLen3 * np.sqrt(1/conOrt)
                        
            SolvedModes.append([mod1,mod2,mod3])
        
        self.SolvedModes = SolvedModes 
          
# =============================================================================
# Axially loaded prebuckled beam - applied axial force with spring
# =============================================================================

class AxialPreBuckledBeam3Spring(Modes):
    def __init__(self,input):
        super().__init__(input) #Inheritance
        
        hs = self.pars['hs']; bs = self.pars['bs']; Es = self.pars['Es']
        hp = self.pars['hp']; bp = self.pars['bp']; Ep = self.pars['Ep']
        L1 = self.pars['L1']; L2 = self.pars['L2']; L3 = self.pars['L3']
        
        rhos = self.pars['rhos']; rhop = self.pars['rhop']; P = self.pars['P']
        k = self.pars['k']
        
        e31 = self.pars['d31']*self.pars['Ep']; e311 = self.pars['e311']
        c111p = self.pars['c111p'] 
  
        self.geoDef = {
            'rhoA1': bs*hs*rhos,
            'rhoA2': ( bp*hp*rhop + bs*hs*rhos ),
            'rhoA3': bs*hs*rhos,
            'EI1': 1/12*bs*Es*(hs)**(3),
            'EI2': 1/12*((bp*Ep*hp + bs*Es*hs ))**(-1) * ((bp)**(2)*(Ep)**(2)*(hp)**(4) + ((bs)**(2)*(Es)**(2)*(hs)**(4) + 2*bp*bs*Ep*Es*hp*hs*(2*(hp)**(2) + (3*hp*hs + 2*(hs)**(2))))),
            'EI3': 1/12*bs*Es*(hs)**(3),
            
            'EA1': bs*hs*Es,
            'EA2': bs*hs*Es + bp*hp*Ep,
            'EA3': bs*hs*Es,
        
            'L1': L1, 'L2': L2, 'L3': L3,
            'P': P,
            'k': k,
            
            'EAn': bp * c111p * hp,
            'EBn': 1/2 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'EIn': 1/12 * bp * c111p * hp * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'EJn': 1/8 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -3 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 2 * ( hp )**( 2 ) + ( 2 * hp * hs + ( hs )**( 2 ) ) ) ) ),
            
            'theta0': bp * e31,
            'theta1': 1/2 * bp * bs * Es * e31 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'theta2': 1/12 * bp * e31 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'psi0': bp * e311,
            'psi1': 1/2 * bp * bs * Es * e311 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'psi2': 1/12 * bp * e311 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            
            'chi': 1/2 * bp * bs * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ) * ( -1 * Es * rhop + Ep * rhos ),
            
            'alpha': bs * Es * hs * ( bp * Ep * hp + bs * Es * hs ) * ( ( bp * Ep * hp * ( bs * Es * hs + k * ( L1 + L3 ) ) + bs * Es * hs * ( bs * Es * hs + k * ( L1 + ( L2 + L3 ) ) ) ) )**( -1 )
            }
        
        self.SolveModes()
        self.SolveDerivates()
        self.PlotModes()
        
    def MatrixFreq(self,omega):
        geoDef = self.geoDef 
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
    
        beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
        beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
        beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
    
        s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )

        
        MatFreq = np.zeros([12,12]) #Initial empty matrix

        #Coefficients
        MatFreq[0,0] = 1
        MatFreq[0,1] = 0
        MatFreq[0,2] = 1
        MatFreq[0,3] = 0
        MatFreq[0,4] = 0
        MatFreq[0,5] = 0
        MatFreq[0,6] = 0
        MatFreq[0,7] = 0
        MatFreq[0,8] = 0
        MatFreq[0,9] = 0
        MatFreq[0,10] = 0
        MatFreq[0,11] = 0
    
        MatFreq[1,0] = 0
        MatFreq[1,1] = s11
        MatFreq[1,2] = 0
        MatFreq[1,3] = s12
        MatFreq[1,4] = 0
        MatFreq[1,5] = 0
        MatFreq[1,6] = 0
        MatFreq[1,7] = 0
        MatFreq[1,8] = 0
        MatFreq[1,9] = 0
        MatFreq[1,10] = 0
        MatFreq[1,11] = 0
    
        MatFreq[2,0] = np.cosh( geoDef['L1'] * s11 )
        MatFreq[2,1] = np.sinh( geoDef['L1'] * s11 )
        MatFreq[2,2] = np.cos( geoDef['L1'] * s12 )
        MatFreq[2,3] = np.sin( geoDef['L1'] * s12 )
        MatFreq[2,4] = -1 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[2,5] = -1 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[2,6] = -1 * np.cos( geoDef['L1'] * s22 )
        MatFreq[2,7] = -1 * np.sin( geoDef['L1'] * s22 )
        MatFreq[2,8] = 0
        MatFreq[2,9] = 0
        MatFreq[2,10] = 0
        MatFreq[2,11] = 0
    
        MatFreq[3,0] = s11 * np.sinh( geoDef['L1'] * s11 )
        MatFreq[3,1] = s11 * np.cosh( geoDef['L1'] * s11 )
        MatFreq[3,2] = -1 * s12 * np.sin( geoDef['L1'] * s12 )
        MatFreq[3,3] = s12 * np.cos( geoDef['L1'] * s12 )
        MatFreq[3,4] = -1 * s21 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[3,5] = -1 * s21 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[3,6] = s22 * np.sin( geoDef['L1'] * s22 )
        MatFreq[3,7] = -1 * s22 * np.cos( geoDef['L1'] * s22 )
        MatFreq[3,8] = 0
        MatFreq[3,9] = 0
        MatFreq[3,10] = 0
        MatFreq[3,11] = 0
    
        MatFreq[4,0] = geoDef['EI1'] * ( s11 )**( 2 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[4,1] = geoDef['EI1'] * ( s11 )**( 2 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[4,2] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[4,3] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[4,4] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[4,5] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[4,6] = geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[4,7] = geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[4,8] = 0
        MatFreq[4,9] = 0
        MatFreq[4,10] = 0
        MatFreq[4,11] = 0
    
        MatFreq[5,0] = geoDef['EI1'] * ( s11 )**( 3 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[5,1] = geoDef['EI1'] * ( s11 )**( 3 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[5,2] = geoDef['EI1'] * ( s12 )**( 3 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[5,3] = -1 * geoDef['EI1'] * ( s12 )**( 3 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[5,4] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[5,5] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[5,6] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[5,7] = geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[5,8] = 0
        MatFreq[5,9] = 0
        MatFreq[5,10] = 0
        MatFreq[5,11] = 0
    
        MatFreq[6,0] = 0
        MatFreq[6,1] = 0
        MatFreq[6,2] = 0
        MatFreq[6,3] = 0
        MatFreq[6,4] = np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,5] = np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,6] = np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,7] = np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,8] = -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,9] = -1 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,10] = -1 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[6,11] = -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[7,0] = 0
        MatFreq[7,1] = 0
        MatFreq[7,2] = 0
        MatFreq[7,3] = 0
        MatFreq[7,4] = s21 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,5] = s21 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,6] = -1 * s22 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,7] = s22 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,8] = -1 * s31 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,9] = -1 * s31 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,10] = s32 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[7,11] = -1 * s32 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[8,0] = 0
        MatFreq[8,1] = 0
        MatFreq[8,2] = 0
        MatFreq[8,3] = 0
        MatFreq[8,4] = geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,5] = geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,6] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,7] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,8] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,9] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,10] = geoDef['EI3'] * ( s32 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[8,11] = geoDef['EI3'] * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[9,0] = 0
        MatFreq[9,1] = 0
        MatFreq[9,2] = 0
        MatFreq[9,3] = 0
        MatFreq[9,4] = geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,5] = geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,6] = geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,7] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,8] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,9] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,10] = -1 * geoDef['EI3'] * ( s32 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[9,11] = geoDef['EI3'] * ( s32 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
    
        MatFreq[10,0] = 0
        MatFreq[10,1] = 0
        MatFreq[10,2] = 0
        MatFreq[10,3] = 0
        MatFreq[10,4] = 0
        MatFreq[10,5] = 0
        MatFreq[10,6] = 0
        MatFreq[10,7] = 0
        MatFreq[10,8] = np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,9] = np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,10] = np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[10,11] = np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
    
        MatFreq[11,0] = 0
        MatFreq[11,1] = s11
        MatFreq[11,2] = 0
        MatFreq[11,3] = s12
        MatFreq[11,4] = 0
        MatFreq[11,5] = 0
        MatFreq[11,6] = 0
        MatFreq[11,7] = 0
        MatFreq[11,8] = s31 * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,9] = s31 * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,10] = -1 * s32 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[11,11] = s32 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        
        return MatFreq
        
    def SolveModes(self):
        
        self.FindNaturalFreq()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution']
             
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation of the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)
        x3 = np.linspace(geoDef['L1'] + geoDef['L2'] ,geoDef['L1'] + geoDef['L2'] + geoDef['L3'] , num = modeRes)
        
        self.xcor = [x1,x2,x3]
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        
        SolvedModes = []
        
        for idx,omega in enumerate(self.NaturalFreq):          
            beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
            beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
            beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
        
            s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )

            C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12 = nf.SolveLinSys( self.MatrixFreq(omega) )
            
            shapeLen1 = lambda x: ( C3 * np.cos( s12 * x ) + ( C1 * np.cosh( s11 * x ) + ( C4 * np.sin( s12 * x ) + C2 * np.sinh( s11 * x ) ) ) )
            shapeLen2 = lambda x: ( C7 * np.cos( s22 * x ) + ( C5 * np.cosh( s21 * x ) + ( C8 * np.sin( s22 * x ) + C6 * np.sinh( s21 * x ) ) ) )
            shapeLen3 = lambda x: ( C11 * np.cos( s32 * x ) + ( C9 * np.cosh( s31 * x ) + ( C12 * np.sin( s32 * x ) + C10 * np.sinh( s31 * x ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1)
            evalShapeLen2 = shapeLen2(x2)
            evalShapeLen3 = shapeLen3(x3)
            
            conOrt = integrate.trapz(geoDef['rhoA1']*evalShapeLen1**2,x1) + \
                     integrate.trapz(geoDef['rhoA2']*evalShapeLen2**2,x2) + \
                      integrate.trapz(geoDef['rhoA3']*evalShapeLen3**2,x3)
            
            mod1 = evalShapeLen1 * np.sqrt(1/conOrt)
            mod2 = evalShapeLen2 * np.sqrt(1/conOrt)
            mod3 = evalShapeLen3 * np.sqrt(1/conOrt)
                        
            SolvedModes.append([mod1,mod2,mod3])
        
        self.SolvedModes = SolvedModes 

# =============================================================================
# Axially loaded postbuckled beam - applied axial force with spring
# =============================================================================

class AxialPostBuckledBeam3Spring(Modes):
    def __init__(self,input):
        super().__init__(input) #Inheritance
        self.bucklingConf = input["bucklingConf"]
        
        hs = self.pars['hs']; bs = self.pars['bs']; Es = self.pars['Es']
        hp = self.pars['hp']; bp = self.pars['bp']; Ep = self.pars['Ep']
        L1 = self.pars['L1']; L2 = self.pars['L2']; L3 = self.pars['L3']
        
        rhos = self.pars['rhos']; rhop = self.pars['rhop']; P = self.pars['P']
        k = self.pars['k']
        
        e31 = self.pars['d31']*self.pars['Ep']; e311 = self.pars['e311']
        c111p = self.pars['c111p'] 
  
        self.geoDef = {
            'rhoA1': bs*hs*rhos,
            'rhoA2': ( bp*hp*rhop + bs*hs*rhos ),
            'rhoA3': bs*hs*rhos,
            'EI1': 1/12*bs*Es*(hs)**(3),
            'EI2': 1/12*((bp*Ep*hp + bs*Es*hs ))**(-1) * ((bp)**(2)*(Ep)**(2)*(hp)**(4) + ((bs)**(2)*(Es)**(2)*(hs)**(4) + 2*bp*bs*Ep*Es*hp*hs*(2*(hp)**(2) + (3*hp*hs + 2*(hs)**(2))))),
            'EI3': 1/12*bs*Es*(hs)**(3),
            
            'EA1': bs*hs*Es,
            'EA2': bs*hs*Es + bp*hp*Ep,
            'EA3': bs*hs*Es,
        
            'L1': L1, 'L2': L2, 'L3': L3,
            'P': P,
            'k': k,
            
            'EAn': bp * c111p * hp,
            'EBn': 1/2 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'EIn': 1/12 * bp * c111p * hp * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'EJn': 1/8 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -3 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 2 * ( hp )**( 2 ) + ( 2 * hp * hs + ( hs )**( 2 ) ) ) ) ),
            
            'theta0': bp * e31,
            'theta1': 1/2 * bp * bs * Es * e31 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'theta2': 1/12 * bp * e31 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'psi0': bp * e311,
            'psi1': 1/2 * bp * bs * Es * e311 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'psi2': 1/12 * bp * e311 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            
            'chi': 1/2 * bp * bs * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ) * ( -1 * Es * rhop + Ep * rhos ),
            
            'alpha': bs * Es * hs * ( bp * Ep * hp + bs * Es * hs ) * ( ( bp * Ep * hp * ( bs * Es * hs + k * ( L1 + L3 ) ) + bs * Es * hs * ( bs * Es * hs + k * ( L1 + ( L2 + L3 ) ) ) ) )**( -1 )
            }

        self.SolveBuckModes()
        self.PlotBuckModes()
        self.SolveModes()
        self.SolveDerivates()
        self.SolveDerivativesBuck()
        self.PlotModes()
        
    def MatrixBuck(self,Phat):
        geoDef = self.geoDef 
        
        lambda1 = (( geoDef['EI1'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
        lambda2 = (( geoDef['EI2'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
        lambda3 = (( geoDef['EI3'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
            
        MatBuck = np.zeros([12,12]) #Initial empty matrix

        #Coefficients
        MatBuck[0,0] = 0
        MatBuck[0,1] = 1
        MatBuck[0,2] = 0
        MatBuck[0,3] = 1
        MatBuck[0,4] = 0
        MatBuck[0,5] = 0
        MatBuck[0,6] = 0
        MatBuck[0,7] = 0
        MatBuck[0,8] = 0
        MatBuck[0,9] = 0
        MatBuck[0,10] = 0
        MatBuck[0,11] = 0
        
        MatBuck[1,0] = lambda1
        MatBuck[1,1] = 0
        MatBuck[1,2] = 1
        MatBuck[1,3] = 0
        MatBuck[1,4] = 0
        MatBuck[1,5] = 0
        MatBuck[1,6] = 0
        MatBuck[1,7] = 0
        MatBuck[1,8] = 0
        MatBuck[1,9] = 0
        MatBuck[1,10] = 0
        MatBuck[1,11] = 0
        
        MatBuck[2,0] = np.sin( geoDef["L1"] * lambda1 )
        MatBuck[2,1] = np.cos( geoDef["L1"] * lambda1 )
        MatBuck[2,2] = geoDef["L1"]
        MatBuck[2,3] = 1
        MatBuck[2,4] = -1 * np.sin( geoDef["L1"] * lambda2 )
        MatBuck[2,5] = -1 * np.cos( geoDef["L1"] * lambda2 )
        MatBuck[2,6] = -1 * geoDef["L1"]
        MatBuck[2,7] = -1
        MatBuck[2,8] = 0
        MatBuck[2,9] = 0
        MatBuck[2,10] = 0
        MatBuck[2,11] = 0
        
        MatBuck[3,0] = lambda1 * np.cos( geoDef["L1"] * lambda1 )
        MatBuck[3,1] = -1 * lambda1 * np.sin( geoDef["L1"] * lambda1 )
        MatBuck[3,2] = 1
        MatBuck[3,3] = 0
        MatBuck[3,4] = -1 * lambda2 * np.cos( geoDef["L1"] * lambda2 )
        MatBuck[3,5] = lambda2 * np.sin( geoDef["L1"] * lambda2 )
        MatBuck[3,6] = -1
        MatBuck[3,7] = 0
        MatBuck[3,8] = 0
        MatBuck[3,9] = 0
        MatBuck[3,10] = 0
        MatBuck[3,11] = 0
        
        MatBuck[4,0] = -1 * geoDef["EI1"] * ( lambda1 )**( 2 ) * np.sin( geoDef["L1"] * lambda1 )
        MatBuck[4,1] = -1 * geoDef["EI1"] * ( lambda1 )**( 2 ) * np.cos( geoDef["L1"] * lambda1 )
        MatBuck[4,2] = 0
        MatBuck[4,3] = 0
        MatBuck[4,4] = geoDef["EI2"] * ( lambda2 )**( 2 ) * np.sin( geoDef["L1"] * lambda2 )
        MatBuck[4,5] = geoDef["EI2"] * ( lambda2 )**( 2 ) * np.cos( geoDef["L1"] * lambda2 )
        MatBuck[4,6] = 0
        MatBuck[4,7] = 0
        MatBuck[4,8] = 0
        MatBuck[4,9] = 0
        MatBuck[4,10] = 0
        MatBuck[4,11] = 0
        
        MatBuck[5,0] = -1 * geoDef["EI1"] * ( lambda1 )**( 3 ) * np.cos( geoDef["L1"] * lambda1 )
        MatBuck[5,1] = geoDef["EI1"] * ( lambda1 )**( 3 ) * np.sin( geoDef["L1"] * lambda1 )
        MatBuck[5,2] = 0
        MatBuck[5,3] = 0
        MatBuck[5,4] = geoDef["EI2"] * ( lambda2 )**( 3 ) * np.cos( geoDef["L1"] * lambda2 )
        MatBuck[5,5] = -1 * geoDef["EI2"] * ( lambda2 )**( 3 ) * np.sin( geoDef["L1"] * lambda2 )
        MatBuck[5,6] = 0
        MatBuck[5,7] = 0
        MatBuck[5,8] = 0
        MatBuck[5,9] = 0
        MatBuck[5,10] = 0
        MatBuck[5,11] = 0
        
        MatBuck[6,0] = 0
        MatBuck[6,1] = 0
        MatBuck[6,2] = 0
        MatBuck[6,3] = 0
        MatBuck[6,4] = np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[6,5] = np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[6,6] = ( geoDef["L1"] + geoDef["L2"] )
        MatBuck[6,7] = 1
        MatBuck[6,8] = -1 * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[6,9] = -1 * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[6,10] = ( -1 * geoDef["L1"] + -1 * geoDef["L2"] )
        MatBuck[6,11] = -1
        
        MatBuck[7,0] = 0
        MatBuck[7,1] = 0
        MatBuck[7,2] = 0
        MatBuck[7,3] = 0
        MatBuck[7,4] = lambda2 * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[7,5] = -1 * lambda2 * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[7,6] = 1
        MatBuck[7,7] = 0
        MatBuck[7,8] = -1 * lambda3 * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[7,9] = lambda3 * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[7,10] = -1
        MatBuck[7,11] = 0
        
        MatBuck[8,0] = 0
        MatBuck[8,1] = 0
        MatBuck[8,2] = 0
        MatBuck[8,3] = 0
        MatBuck[8,4] = -1 * geoDef["EI2"] * ( lambda2 )**( 2 ) * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[8,5] = -1 * geoDef["EI2"] * ( lambda2 )**( 2 ) * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[8,6] = 0
        MatBuck[8,7] = 0
        MatBuck[8,8] = geoDef["EI3"] * ( lambda3 )**( 2 ) * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[8,9] = geoDef["EI3"] * ( lambda3 )**( 2 ) * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[8,10] = 0
        MatBuck[8,11] = 0
        
        MatBuck[9,0] = 0
        MatBuck[9,1] = 0
        MatBuck[9,2] = 0
        MatBuck[9,3] = 0
        MatBuck[9,4] = -1 * geoDef["EI2"] * ( lambda2 )**( 3 ) * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[9,5] = geoDef["EI2"] * ( lambda2 )**( 3 ) * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda2 )
        MatBuck[9,6] = 0
        MatBuck[9,7] = 0
        MatBuck[9,8] = geoDef["EI3"] * ( lambda3 )**( 3 ) * np.cos( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[9,9] = -1 * geoDef["EI3"] * ( lambda3 )**( 3 ) * np.sin( ( geoDef["L1"] + geoDef["L2"] ) * lambda3 )
        MatBuck[9,10] = 0
        MatBuck[9,11] = 0
        
        MatBuck[10,0] = 0
        MatBuck[10,1] = 0
        MatBuck[10,2] = 0
        MatBuck[10,3] = 0
        MatBuck[10,4] = 0
        MatBuck[10,5] = 0
        MatBuck[10,6] = 0
        MatBuck[10,7] = 0
        MatBuck[10,8] = np.sin( ( geoDef["L1"] + ( geoDef["L2"] + geoDef["L3"] ) ) * lambda3 )
        MatBuck[10,9] = np.cos( ( geoDef["L1"] + ( geoDef["L2"] + geoDef["L3"] ) ) * lambda3 )
        MatBuck[10,10] = ( geoDef["L1"] + ( geoDef["L2"] + geoDef["L3"] ) )
        MatBuck[10,11] = 1
        
        MatBuck[11,0] = 0
        MatBuck[11,1] = 0
        MatBuck[11,2] = 0
        MatBuck[11,3] = 0
        MatBuck[11,4] = 0
        MatBuck[11,5] = 0
        MatBuck[11,6] = 0
        MatBuck[11,7] = 0
        MatBuck[11,8] = lambda3 * np.cos( ( geoDef["L1"] + ( geoDef["L2"] + geoDef["L3"] ) ) * lambda3 )
        MatBuck[11,9] = -1 * lambda3 * np.sin( ( geoDef["L1"] + ( geoDef["L2"] + geoDef["L3"] ) ) * lambda3 )
        MatBuck[11,10] = 1
        MatBuck[11,11] = 0
        
        return MatBuck
    
    def SolveBuckModes(self):
        
        self.FindCriticalLoad()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution'] 
        #resolution of the buckled modes has to be the as the mode shapes in order to perform the dot product between vectors for the modal parameters
             
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation for the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)
        x3 = np.linspace(geoDef['L1'] + geoDef['L2'] ,geoDef['L1'] + geoDef['L2'] + geoDef['L3'] , num = modeRes)
        
        self.xcorbuck = [x1,x2,x3]
        
        SolvedBuckModes = []; Kvalues = []; lambdaValues = []; LambdaQ = []

        for idx,Phat in enumerate(self.Pcrit):          
            lambda1 = (( geoDef['EI1'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
            lambda2 = (( geoDef['EI2'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
            lambda3 = (( geoDef['EI3'] )**( -1 ) * Phat * geoDef['alpha'] )**( 1/2 )
            
            K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12 = nf.SolveLinSys( self.MatrixBuck(Phat) )

            shapeLen1 = lambda x: ( K4 + ( K3 * x + ( K2 * np.cos( x * lambda1 ) + K1 * np.sin( x * lambda1 ) ) ) )
            shapeLen2 = lambda x: ( K8 + ( K7 * x + ( K6 * np.cos( x * lambda2 ) + K5 * np.sin( x * lambda2 ) ) ) )
            shapeLen3 = lambda x: ( K12 + ( K11 * x + ( K10 * np.cos( x * lambda3 ) + K9 * np.sin( x * lambda3 ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1); dfmode1 = nf.NumDer(x1,evalShapeLen1,1); 
            evalShapeLen2 = shapeLen2(x2); dfmode2 = nf.NumDer(x2,evalShapeLen2,1); 
            evalShapeLen3 = shapeLen3(x3); dfmode3 = nf.NumDer(x3,evalShapeLen3,1);
            
            Gamma = 0.5 * self.pars['k'] * (\
                    integrate.trapz(dfmode1**2,x1) +\
                    integrate.trapz(dfmode2**2,x2) +\
                    integrate.trapz(dfmode3**2,x3) )
                    
            bamp = abs( ( geoDef["alpha"] )**( -1/2 ) * ( Gamma )**( -1/2 ) * ( ( geoDef["P"] * geoDef["alpha"] + -1 * geoDef["EI1"] * ( lambda1 )**( 2 ) ) )**( 1/2 ) )
            # also works with EI2 & lambda2 and EI3 & lambda3
            
            Kvalues.append([ K1[0]*bamp, K2[0]*bamp , K3[0]*bamp,  K4[0]*bamp, 
                             K5[0]*bamp, K6[0]*bamp , K7[0]*bamp,  K8[0]*bamp, 
                             K9[0]*bamp, K10[0]*bamp, K11[0]*bamp, K12[0]*bamp]  )
            
            lambdaValues.append( [lambda1, lambda2, lambda3] )
            
            mod1 = evalShapeLen1 * bamp
            mod2 = evalShapeLen2 * bamp
            mod3 = evalShapeLen3 * bamp
                        
            SolvedBuckModes.append([mod1,mod2,mod3])
            
            LambdaQ.append( 
                integrate.trapz( nf.NumDer(x1,mod1,1) * nf.NumDer(x1,mod1,3) ,x1) + \
                integrate.trapz( nf.NumDer(x2,mod2,1) * nf.NumDer(x2,mod2,3) ,x2) + \
                integrate.trapz( nf.NumDer(x3,mod3,1) * nf.NumDer(x3,mod3,3) ,x3) )
        
        
        self.SolvedBuckModes = SolvedBuckModes 
        self.Kvalues = Kvalues
        self.lambdaValues = lambdaValues
        self.LambdaQ = LambdaQ

    def MatrixFreq(self,omega):
        geoDef = self.geoDef 
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
   
        
        beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
        beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
        beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
        
        lambda1 = self.lambdaValues[0][0]
        lambda2 = self.lambdaValues[0][1] 
        lambda3 = self.lambdaValues[0][2]
        
        s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
        s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
   
        d1 = ( 2 * geoDef['EI1'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        d2 = ( 2 * geoDef['EI2'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        d3 = ( 2 * geoDef['EI3'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        
        MatFreq = np.zeros([15,15]) #Initial empty matrix
  
        Lambdaq = self.LambdaQ[0]
        
        K1 =  self.Kvalues[0][0]
        K2 =  self.Kvalues[0][1]
        K3 =  self.Kvalues[0][2]
        K4 =  self.Kvalues[0][3]
        K5 =  self.Kvalues[0][4]
        K6 =  self.Kvalues[0][5]
        K7 =  self.Kvalues[0][6]
        K8 =  self.Kvalues[0][7]
        K9 =  self.Kvalues[0][8]
        K10 = self.Kvalues[0][9]
        K11 = self.Kvalues[0][10]
        K12 = self.Kvalues[0][11]
           
        MatFreq[0,0] = 1
        MatFreq[0,1] = 0
        MatFreq[0,2] = 1
        MatFreq[0,3] = 0
        MatFreq[0,4] = -1 * K2 * ( lambda1 )**( 2 )
        MatFreq[0,5] = 0
        MatFreq[0,6] = 0
        MatFreq[0,7] = 0
        MatFreq[0,8] = 0
        MatFreq[0,9] = 0
        MatFreq[0,10] = 0
        MatFreq[0,11] = 0
        MatFreq[0,12] = 0
        MatFreq[0,13] = 0
        MatFreq[0,14] = 0
        
        MatFreq[1,0] = 0
        MatFreq[1,1] = s11
        MatFreq[1,2] = 0
        MatFreq[1,3] = s12
        MatFreq[1,4] = -1 * K1 * ( lambda1 )**( 3 )
        MatFreq[1,5] = 0
        MatFreq[1,6] = 0
        MatFreq[1,7] = 0
        MatFreq[1,8] = 0
        MatFreq[1,9] = 0
        MatFreq[1,10] = 0
        MatFreq[1,11] = 0
        MatFreq[1,12] = 0
        MatFreq[1,13] = 0
        MatFreq[1,14] = 0
        
        MatFreq[2,0] = np.cosh( geoDef['L1'] * s11 )
        MatFreq[2,1] = np.sinh( geoDef['L1'] * s11 )
        MatFreq[2,2] = np.cos( geoDef['L1'] * s12 )
        MatFreq[2,3] = np.sin( geoDef['L1'] * s12 )
        MatFreq[2,4] = -1 * ( lambda1 )**( 2 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) )
        MatFreq[2,5] = -1 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[2,6] = -1 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[2,7] = -1 * np.cos( geoDef['L1'] * s22 )
        MatFreq[2,8] = -1 * np.sin( geoDef['L1'] * s22 )
        MatFreq[2,9] = ( lambda2 )**( 2 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) )
        MatFreq[2,10] = 0
        MatFreq[2,11] = 0
        MatFreq[2,12] = 0
        MatFreq[2,13] = 0
        MatFreq[2,14] = 0
        
        MatFreq[3,0] = s11 * np.sinh( geoDef['L1'] * s11 )
        MatFreq[3,1] = s11 * np.cosh( geoDef['L1'] * s11 )
        MatFreq[3,2] = -1 * s12 * np.sin( geoDef['L1'] * s12 )
        MatFreq[3,3] = s12 * np.cos( geoDef['L1'] * s12 )
        MatFreq[3,4] = ( lambda1 )**( 3 ) * ( -1 * K1 * np.cos( geoDef['L1'] * lambda1 ) + K2 * np.sin( geoDef['L1'] * lambda1 ) )
        MatFreq[3,5] = -1 * s21 * np.sinh( geoDef['L1'] * s21 )
        MatFreq[3,6] = -1 * s21 * np.cosh( geoDef['L1'] * s21 )
        MatFreq[3,7] = s22 * np.sin( geoDef['L1'] * s22 )
        MatFreq[3,8] = -1 * s22 * np.cos( geoDef['L1'] * s22 )
        MatFreq[3,9] = ( lambda2 )**( 3 ) * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) )
        MatFreq[3,10] = 0
        MatFreq[3,11] = 0
        MatFreq[3,12] = 0
        MatFreq[3,13] = 0
        MatFreq[3,14] = 0
        
        MatFreq[4,0] = geoDef['EI1'] * ( s11 )**( 2 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[4,1] = geoDef['EI1'] * ( s11 )**( 2 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[4,2] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[4,3] = -1 * geoDef['EI1'] * ( s12 )**( 2 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[4,4] = geoDef['EI1'] * ( lambda1 )**( 4 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) )
        MatFreq[4,5] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[4,6] = -1 * geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[4,7] = geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[4,8] = geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[4,9] = -1 * geoDef['EI2'] * ( lambda2 )**( 4 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) )
        MatFreq[4,10] = 0
        MatFreq[4,11] = 0
        MatFreq[4,12] = 0
        MatFreq[4,13] = 0
        MatFreq[4,14] = 0
        
        MatFreq[5,0] = geoDef['EI1'] * ( s11 )**( 3 ) * np.sinh( geoDef['L1'] * s11 )
        MatFreq[5,1] = geoDef['EI1'] * ( s11 )**( 3 ) * np.cosh( geoDef['L1'] * s11 )
        MatFreq[5,2] = geoDef['EI1'] * ( s12 )**( 3 ) * np.sin( geoDef['L1'] * s12 )
        MatFreq[5,3] = -1 * geoDef['EI1'] * ( s12 )**( 3 ) * np.cos( geoDef['L1'] * s12 )
        MatFreq[5,4] = geoDef['EI1'] * ( lambda1 )**( 5 ) * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) )
        MatFreq[5,5] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( geoDef['L1'] * s21 )
        MatFreq[5,6] = -1 * geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( geoDef['L1'] * s21 )
        MatFreq[5,7] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( geoDef['L1'] * s22 )
        MatFreq[5,8] = geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( geoDef['L1'] * s22 )
        MatFreq[5,9] = geoDef['EI2'] * ( lambda2 )**( 5 ) * ( -1 * K5 * np.cos( geoDef['L1'] * lambda2 ) + K6 * np.sin( geoDef['L1'] * lambda2 ) )
        MatFreq[5,10] = 0
        MatFreq[5,11] = 0
        MatFreq[5,12] = 0
        MatFreq[5,13] = 0
        MatFreq[5,14] = 0
        
        MatFreq[6,0] = 0
        MatFreq[6,1] = 0
        MatFreq[6,2] = 0
        MatFreq[6,3] = 0
        MatFreq[6,4] = 0
        MatFreq[6,5] = np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,6] = np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[6,7] = np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,8] = np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[6,9] = -1 * ( lambda2 )**( 2 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) )
        MatFreq[6,10] = -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,11] = -1 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[6,12] = -1 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[6,13] = -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[6,14] = ( lambda3 )**( 2 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) )
        
        MatFreq[7,0] = 0
        MatFreq[7,1] = 0
        MatFreq[7,2] = 0
        MatFreq[7,3] = 0
        MatFreq[7,4] = 0
        MatFreq[7,5] = s21 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,6] = s21 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[7,7] = -1 * s22 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,8] = s22 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[7,9] = ( lambda2 )**( 3 ) * ( -1 * K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) )
        MatFreq[7,10] = -1 * s31 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,11] = -1 * s31 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[7,12] = s32 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[7,13] = -1 * s32 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[7,14] = ( lambda3 )**( 3 ) * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) )
        
        MatFreq[8,0] = 0
        MatFreq[8,1] = 0
        MatFreq[8,2] = 0
        MatFreq[8,3] = 0
        MatFreq[8,4] = 0
        MatFreq[8,5] = geoDef['EI2'] * ( s21 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,6] = geoDef['EI2'] * ( s21 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[8,7] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,8] = -1 * geoDef['EI2'] * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[8,9] = geoDef['EI2'] * ( lambda2 )**( 4 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) )
        MatFreq[8,10] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,11] = -1 * geoDef['EI3'] * ( s31 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[8,12] = geoDef['EI3'] * ( s32 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[8,13] = geoDef['EI3'] * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[8,14] = -1 * geoDef['EI3'] * ( lambda3 )**( 4 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) )
        
        MatFreq[9,0] = 0
        MatFreq[9,1] = 0
        MatFreq[9,2] = 0
        MatFreq[9,3] = 0
        MatFreq[9,4] = 0
        MatFreq[9,5] = geoDef['EI2'] * ( s21 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,6] = geoDef['EI2'] * ( s21 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 )
        MatFreq[9,7] = geoDef['EI2'] * ( s22 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,8] = -1 * geoDef['EI2'] * ( s22 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 )
        MatFreq[9,9] = geoDef['EI2'] * ( lambda2 )**( 5 ) * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) )
        MatFreq[9,10] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,11] = -1 * geoDef['EI3'] * ( s31 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 )
        MatFreq[9,12] = -1 * geoDef['EI3'] * ( s32 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[9,13] = geoDef['EI3'] * ( s32 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 )
        MatFreq[9,14] = geoDef['EI3'] * ( lambda3 )**( 5 ) * ( -1 * K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) )
        
        MatFreq[10,0] = 0
        MatFreq[10,1] = 0
        MatFreq[10,2] = 0
        MatFreq[10,3] = 0
        MatFreq[10,4] = 0
        MatFreq[10,5] = 0
        MatFreq[10,6] = 0
        MatFreq[10,7] = 0
        MatFreq[10,8] = 0
        MatFreq[10,9] = 0
        MatFreq[10,10] = np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,11] = np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[10,12] = np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[10,13] = np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[10,14] = -1 * ( lambda3 )**( 2 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) )
        
        MatFreq[11,0] = 0
        MatFreq[11,1] = 0
        MatFreq[11,2] = 0
        MatFreq[11,3] = 0
        MatFreq[11,4] = 0
        MatFreq[11,5] = 0
        MatFreq[11,6] = 0
        MatFreq[11,7] = 0
        MatFreq[11,8] = 0
        MatFreq[11,9] = 0
        MatFreq[11,10] = s31 * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,11] = s31 * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 )
        MatFreq[11,12] = -1 * s32 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[11,13] = s32 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 )
        MatFreq[11,14] = ( lambda3 )**( 3 ) * ( -1 * K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) )
        
        MatFreq[12,0] = d1 * ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( -1 * K1 * ( s11 )**( 2 ) * lambda1 + ( -1 * K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( np.cosh( geoDef['L1'] * s11 ) * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) + s11 * ( lambda1 )**( 2 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) * np.sinh( geoDef['L1'] * s11 ) ) ) )
        MatFreq[12,1] = ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( d1 * s11 * ( lambda1 )**( 2 ) * ( -1 * K2 + np.cosh( geoDef['L1'] * s11 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) ) + d1 * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) * np.sinh( geoDef['L1'] * s11 ) )
        MatFreq[12,2] = d1 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( -1 * K1 * ( s12 )**( 2 ) * lambda1 + ( K3 * ( -1 * ( s12 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s12 * ( lambda1 )**( 2 ) * np.sin( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.cos( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) ) )
        MatFreq[12,3] = d1 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( K2 * s12 * ( lambda1 )**( 2 ) + ( -1 * s12 * ( lambda1 )**( 2 ) * np.cos( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.sin( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) )
        MatFreq[12,4] = ( ( beta1 )**( 4 ) + d1 * Lambdaq )
        MatFreq[12,5] = d1 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( geoDef['L1'] * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + s21 * ( lambda2 )**( 2 ) * ( -1 * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[12,6] = d1 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * s21 * ( lambda2 )**( 2 ) * np.cosh( geoDef['L1'] * s21 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( s21 * ( lambda2 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) + ( -1 * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[12,7] = d1 * ( ( s22 + -1 * lambda2 ) )**( -1 ) * ( ( s22 + lambda2 ) )**( -1 ) * ( np.cos( geoDef['L1'] * s22 ) * ( K7 * ( -1 * ( s22 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s22 )**( 2 ) * lambda2 * ( -1 * K5 * np.cos( geoDef['L1'] * lambda2 ) + K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( s22 * ( lambda2 )**( 2 ) * ( -1 * np.sin( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K7 * ( s22 + -1 * lambda2 ) * ( s22 + lambda2 ) + ( s22 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) )
        MatFreq[12,8] = d1 * ( ( ( s22 )**( 2 ) + -1 * ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * K7 * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( lambda2 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( -1 * K5 * ( s22 )**( 2 ) * lambda2 * np.cos( geoDef['L1'] * lambda2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( -1 * K7 * ( lambda2 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K5 * ( s22 )**( 2 ) * lambda2 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K6 * ( s22 )**( 2 ) * lambda2 * np.sin( geoDef['L1'] * s22 ) * np.sin( geoDef['L1'] * lambda2 ) + ( s22 * ( lambda2 )**( 2 ) * np.cos( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( -1 * K6 * ( s22 )**( 2 ) * lambda2 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * s22 * ( lambda2 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[12,9] = 0
        MatFreq[12,10] = d1 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s31 * ( lambda3 )**( 2 ) * ( -1 * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[12,11] = d1 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) + ( -1 * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[12,12] = d1 * ( ( s32 + -1 * lambda3 ) )**( -1 ) * ( ( s32 + lambda3 ) )**( -1 ) * ( np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K11 * ( -1 * ( s32 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s32 )**( 2 ) * lambda3 * ( -1 * K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K11 * ( s32 + -1 * lambda3 ) * ( s32 + lambda3 ) + ( s32 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s32 * ( lambda3 )**( 2 ) * ( -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) )
        MatFreq[12,13] = d1 * ( ( ( s32 )**( 2 ) + -1 * ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( -1 * K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( -1 * K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + ( s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( -1 * K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[12,14] = 0
        
        MatFreq[13,0] = d2 * ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( -1 * K1 * ( s11 )**( 2 ) * lambda1 + ( -1 * K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( np.cosh( geoDef['L1'] * s11 ) * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) + s11 * ( lambda1 )**( 2 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) * np.sinh( geoDef['L1'] * s11 ) ) ) )
        MatFreq[13,1] = ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( d2 * s11 * ( lambda1 )**( 2 ) * ( -1 * K2 + np.cosh( geoDef['L1'] * s11 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) ) + d2 * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) * np.sinh( geoDef['L1'] * s11 ) )
        MatFreq[13,2] = d2 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( -1 * K1 * ( s12 )**( 2 ) * lambda1 + ( K3 * ( -1 * ( s12 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s12 * ( lambda1 )**( 2 ) * np.sin( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.cos( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) ) )
        MatFreq[13,3] = d2 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( K2 * s12 * ( lambda1 )**( 2 ) + ( -1 * s12 * ( lambda1 )**( 2 ) * np.cos( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.sin( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) )
        MatFreq[13,4] = 0
        MatFreq[13,5] = d2 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( geoDef['L1'] * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + s21 * ( lambda2 )**( 2 ) * ( -1 * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[13,6] = d2 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * s21 * ( lambda2 )**( 2 ) * np.cosh( geoDef['L1'] * s21 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( s21 * ( lambda2 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) + ( -1 * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[13,7] = d2 * ( ( s22 + -1 * lambda2 ) )**( -1 ) * ( ( s22 + lambda2 ) )**( -1 ) * ( np.cos( geoDef['L1'] * s22 ) * ( K7 * ( -1 * ( s22 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s22 )**( 2 ) * lambda2 * ( -1 * K5 * np.cos( geoDef['L1'] * lambda2 ) + K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( s22 * ( lambda2 )**( 2 ) * ( -1 * np.sin( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K7 * ( s22 + -1 * lambda2 ) * ( s22 + lambda2 ) + ( s22 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) )
        MatFreq[13,8] = d2 * ( ( ( s22 )**( 2 ) + -1 * ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * K7 * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( lambda2 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( -1 * K5 * ( s22 )**( 2 ) * lambda2 * np.cos( geoDef['L1'] * lambda2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( -1 * K7 * ( lambda2 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K5 * ( s22 )**( 2 ) * lambda2 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K6 * ( s22 )**( 2 ) * lambda2 * np.sin( geoDef['L1'] * s22 ) * np.sin( geoDef['L1'] * lambda2 ) + ( s22 * ( lambda2 )**( 2 ) * np.cos( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( -1 * K6 * ( s22 )**( 2 ) * lambda2 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * s22 * ( lambda2 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[13,9] = ( ( beta2 )**( 4 ) + d2 * Lambdaq )
        MatFreq[13,10] = d2 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s31 * ( lambda3 )**( 2 ) * ( -1 * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[13,11] = d2 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) + ( -1 * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[13,12] = d2 * ( ( s32 + -1 * lambda3 ) )**( -1 ) * ( ( s32 + lambda3 ) )**( -1 ) * ( np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K11 * ( -1 * ( s32 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s32 )**( 2 ) * lambda3 * ( -1 * K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K11 * ( s32 + -1 * lambda3 ) * ( s32 + lambda3 ) + ( s32 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s32 * ( lambda3 )**( 2 ) * ( -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) )
        MatFreq[13,13] = d2 * ( ( ( s32 )**( 2 ) + -1 * ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( -1 * K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( -1 * K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + ( s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( -1 * K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[13,14] = 0
        
        MatFreq[14,0] = d3 * ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( -1 * K1 * ( s11 )**( 2 ) * lambda1 + ( -1 * K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( np.cosh( geoDef['L1'] * s11 ) * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) + s11 * ( lambda1 )**( 2 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) * np.sinh( geoDef['L1'] * s11 ) ) ) )
        MatFreq[14,1] = ( ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) )**( -1 ) * ( d3 * s11 * ( lambda1 )**( 2 ) * ( -1 * K2 + np.cosh( geoDef['L1'] * s11 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) ) + d3 * ( K3 * ( ( s11 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s11 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) * np.sinh( geoDef['L1'] * s11 ) )
        MatFreq[14,2] = d3 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( -1 * K1 * ( s12 )**( 2 ) * lambda1 + ( K3 * ( -1 * ( s12 )**( 2 ) + ( lambda1 )**( 2 ) ) + ( s12 * ( lambda1 )**( 2 ) * np.sin( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.cos( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) ) )
        MatFreq[14,3] = d3 * ( ( s12 + -1 * lambda1 ) )**( -1 ) * ( ( s12 + lambda1 ) )**( -1 ) * ( K2 * s12 * ( lambda1 )**( 2 ) + ( -1 * s12 * ( lambda1 )**( 2 ) * np.cos( geoDef['L1'] * s12 ) * ( K2 * np.cos( geoDef['L1'] * lambda1 ) + K1 * np.sin( geoDef['L1'] * lambda1 ) ) + np.sin( geoDef['L1'] * s12 ) * ( K3 * ( s12 + -1 * lambda1 ) * ( s12 + lambda1 ) + ( s12 )**( 2 ) * lambda1 * ( K1 * np.cos( geoDef['L1'] * lambda1 ) + -1 * K2 * np.sin( geoDef['L1'] * lambda1 ) ) ) ) )
        MatFreq[14,4] = 0
        MatFreq[14,5] = d3 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( geoDef['L1'] * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + s21 * ( lambda2 )**( 2 ) * ( -1 * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[14,6] = d3 * ( ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * s21 * ( lambda2 )**( 2 ) * np.cosh( geoDef['L1'] * s21 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( s21 * ( lambda2 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) + ( -1 * ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( geoDef['L1'] * lambda2 ) + -1 * K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) * np.sinh( geoDef['L1'] * s21 ) + ( K7 * ( ( s21 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s21 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s21 ) ) ) )
        MatFreq[14,7] = d3 * ( ( s22 + -1 * lambda2 ) )**( -1 ) * ( ( s22 + lambda2 ) )**( -1 ) * ( np.cos( geoDef['L1'] * s22 ) * ( K7 * ( -1 * ( s22 )**( 2 ) + ( lambda2 )**( 2 ) ) + ( s22 )**( 2 ) * lambda2 * ( -1 * K5 * np.cos( geoDef['L1'] * lambda2 ) + K6 * np.sin( geoDef['L1'] * lambda2 ) ) ) + ( s22 * ( lambda2 )**( 2 ) * ( -1 * np.sin( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) + np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K7 * ( s22 + -1 * lambda2 ) * ( s22 + lambda2 ) + ( s22 )**( 2 ) * lambda2 * ( K5 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * K6 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) )
        MatFreq[14,8] = d3 * ( ( ( s22 )**( 2 ) + -1 * ( lambda2 )**( 2 ) ) )**( -1 ) * ( -1 * K7 * ( s22 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( lambda2 )**( 2 ) * np.sin( geoDef['L1'] * s22 ) + ( -1 * K5 * ( s22 )**( 2 ) * lambda2 * np.cos( geoDef['L1'] * lambda2 ) * np.sin( geoDef['L1'] * s22 ) + ( K7 * ( s22 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( -1 * K7 * ( lambda2 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K5 * ( s22 )**( 2 ) * lambda2 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) + ( K6 * ( s22 )**( 2 ) * lambda2 * np.sin( geoDef['L1'] * s22 ) * np.sin( geoDef['L1'] * lambda2 ) + ( s22 * ( lambda2 )**( 2 ) * np.cos( geoDef['L1'] * s22 ) * ( K6 * np.cos( geoDef['L1'] * lambda2 ) + K5 * np.sin( geoDef['L1'] * lambda2 ) ) + ( -1 * K6 * ( s22 )**( 2 ) * lambda2 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + -1 * s22 * ( lambda2 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s22 ) * ( K6 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) + K5 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda2 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[14,9] = 0
        MatFreq[14,10] = d3 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s31 * ( lambda3 )**( 2 ) * ( -1 * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[14,11] = d3 * ( ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( s31 * ( lambda3 )**( 2 ) * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) + ( -1 * ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * s31 ) + ( K11 * ( ( s31 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s31 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s31 ) ) ) )
        MatFreq[14,12] = d3 * ( ( s32 + -1 * lambda3 ) )**( -1 ) * ( ( s32 + lambda3 ) )**( -1 ) * ( np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K11 * ( -1 * ( s32 )**( 2 ) + ( lambda3 )**( 2 ) ) + ( s32 )**( 2 ) * lambda3 * ( -1 * K9 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K10 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) ) + ( np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K11 * ( s32 + -1 * lambda3 ) * ( s32 + lambda3 ) + ( s32 )**( 2 ) * lambda3 * ( K9 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * K10 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) + s32 * ( lambda3 )**( 2 ) * ( -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) )
        MatFreq[14,13] = d3 * ( ( ( s32 )**( 2 ) + -1 * ( lambda3 )**( 2 ) ) )**( -1 ) * ( -1 * K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( -1 * K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) + ( K11 * ( s32 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( -1 * K11 * ( lambda3 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K9 * ( s32 )**( 2 ) * lambda3 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) + ( K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + ( s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * lambda3 ) ) + ( -1 * K10 * ( s32 )**( 2 ) * lambda3 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + -1 * s32 * ( lambda3 )**( 2 ) * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * s32 ) * ( K10 * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) + K9 * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * lambda3 ) ) ) ) ) ) ) ) ) ) )
        MatFreq[14,14] = ( ( beta3 )**( 4 ) + d3 * Lambdaq )
        
        return MatFreq
    
    def SolveModes(self):
        self.FindNaturalFreq()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution']
             
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation of the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)
        x3 = np.linspace(geoDef['L1'] + geoDef['L2'] ,geoDef['L1'] + geoDef['L2'] + geoDef['L3'] , num = modeRes)
        
        self.xcor = [x1,x2,x3]
        
        k1 = (( geoDef['EI1'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k2 = (( geoDef['EI2'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
        k3 = (( geoDef['EI3'] )**(-1) * geoDef['P'] * geoDef['alpha'])**(1/2)
   
        d1 = ( 2* geoDef['EI1'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        d2 = ( 2* geoDef['EI2'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        d3 = ( 2* geoDef['EI3'] )**( -1 ) * geoDef['k'] * geoDef['alpha']
        
        lambda1 = self.lambdaValues[0][0]
        lambda2 = self.lambdaValues[0][1] 
        lambda3 = self.lambdaValues[0][2]
                
        K1 =  self.Kvalues[0][0]
        K2 =  self.Kvalues[0][1]
        K3 =  self.Kvalues[0][2]
        K4 =  self.Kvalues[0][3]
        K5 =  self.Kvalues[0][4]
        K6 =  self.Kvalues[0][5]
        K7 =  self.Kvalues[0][6]
        K8 =  self.Kvalues[0][7]
        K9 =  self.Kvalues[0][8]
        K10 = self.Kvalues[0][9]
        K11 = self.Kvalues[0][10]
        K12 = self.Kvalues[0][11]
        
        SolvedModes = []
        
        for idx,omega in enumerate(self.NaturalFreq):          
            beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
            beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
            beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
            
            s11 = ( ( -1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s12 = ( ( 1/2 * ( k1 )**( 2 ) + ( ( 1/4 * ( k1 )**( 4 ) + ( beta1 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s21 = ( ( -1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s22 = ( ( 1/2 * ( k2 )**( 2 ) + ( ( 1/4 * ( k2 )**( 4 ) + ( beta2 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s31 = ( ( -1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            s32 = ( ( 1/2 * ( k3 )**( 2 ) + ( ( 1/4 * ( k3 )**( 4 ) + ( beta3 )**( 4 ) ) )**( 1/2 ) ) )**( 1/2 )
            
            C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15 = nf.SolveLinSys( self.MatrixFreq(omega) )
            
            shapeLen1 = lambda x: ( C3 * np.cos( s12 * x ) + ( -1 * C5 * K2 * ( lambda1 )**( 2 ) * np.cos( x * lambda1 ) + ( C1 * np.cosh( s11 * x ) + ( C4 * np.sin( s12 * x ) + ( -1 * C5 * K1 * ( lambda1 )**( 2 ) * np.sin( x * lambda1 ) + C2 * np.sinh( s11 * x ) ) ) ) ) )
            shapeLen2 = lambda x: ( C8 * np.cos( s22 * x ) + ( -1 * C10 * K6 * ( lambda2 )**( 2 ) * np.cos( x * lambda2 ) + ( C6 * np.cosh( s21 * x ) + ( C9 * np.sin( s22 * x ) + ( -1 * C10 * K5 * ( lambda2 )**( 2 ) * np.sin( x * lambda2 ) + C7 * np.sinh( s21 * x ) ) ) ) ) )
            shapeLen3 = lambda x: ( C13 * np.cos( s32 * x ) + ( -1 * C15 * K10 * ( lambda3 )**( 2 ) * np.cos( x * lambda3 ) + ( C11 * np.cosh( s31 * x ) + ( C14 * np.sin( s32 * x ) + ( -1 * C15 * K9 * ( lambda3 )**( 2 ) * np.sin( x * lambda3 ) + C12 * np.sinh( s31 * x ) ) ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1)
            evalShapeLen2 = shapeLen2(x2)
            evalShapeLen3 = shapeLen3(x3)
            
            conOrt = integrate.trapz( geoDef['rhoA1'] * evalShapeLen1**2, x1) + \
                     integrate.trapz( geoDef['rhoA2'] * evalShapeLen2**2, x2) + \
                     integrate.trapz( geoDef['rhoA3'] * evalShapeLen3**2, x3)
            
            mod1 = evalShapeLen1 * np.sqrt(1/conOrt)
            mod2 = evalShapeLen2 * np.sqrt(1/conOrt)
            mod3 = evalShapeLen3 * np.sqrt(1/conOrt)
                        
            SolvedModes.append([mod1,mod2,mod3])
        
        self.SolvedModes = SolvedModes 

# =============================================================================
#  Cantilever Beam - 2 Sections
# =============================================================================

class CantileverBeam2(Modes):
    def __init__(self,input):
        Modes.__init__(self,input) #Inheritance

        hs = self.pars['hs']; bs = self.pars['bs']; Es = self.pars['Es']
        hp = self.pars['hp']; bp = self.pars['bp']; Ep = self.pars['Ep']
        L1 = self.pars['L1']; L2 = self.pars['L2']; 
        
        rhos = self.pars['rhos']; rhop = self.pars['rhop'];
        
        e31 = self.pars['d31']*self.pars['Ep']; e311 = self.pars['e311']
        c111p = self.pars['c111p'] 
        
        self.geoDef = {
            'rhoA2': bs*hs*rhos,
            'rhoA1': ( bp*hp*rhop + bs*hs*rhos ),
            'EI2': 1/12*bs*Es*(hs)**(3),
            'EI1': 1/12*((bp*Ep*hp + bs*Es*hs ))**(-1) * ((bp)**(2)*(Ep)**(2)*(hp)**(4) + ((bs)**(2)*(Es)**(2)*(hs)**(4) + 2*bp*bs*Ep*Es*hp*hs*(2*(hp)**(2) + (3*hp*hs + 2*(hs)**(2))))),
            
            'EA1': bs*hs*Es + bp*hp*Ep,
            'EA2': bs*hs*Es,
            
            'L1': L1, 'L2': L2,
            
            'EAn': bp * c111p * hp,
            'EBn': 1/2 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'EIn': 1/12 * bp * c111p * hp * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'EJn': 1/8 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -3 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 2 * ( hp )**( 2 ) + ( 2 * hp * hs + ( hs )**( 2 ) ) ) ) ),
            
            'theta0': bp * e31,
            'theta1': 1/2 * bp * bs * Es * e31 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'theta2': 1/12 * bp * e31 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'psi0': bp * e311,
            'psi1': 1/2 * bp * bs * Es * e311 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'psi2': 1/12 * bp * e311 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            
            'chi': 1/2 * bp * bs * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ) * ( -1 * Es * rhop + Ep * rhos )
         }
        
        self.SolveModes()
        self.SolveDerivates()
        self.PlotModes()
        
    def MatrixFreq(self,omega):
        geoDef = self.geoDef 
        
        beta1 = ( ( geoDef['EI1'] )**( -1 ) * geoDef['rhoA1'] * ( omega )**( 2 ) )**( 1/4 )
        beta2 = ( ( geoDef['EI2'] )**( -1 ) * geoDef['rhoA2'] * ( omega )**( 2 ) )**( 1/4 )

        MatFreq = np.zeros([8,8])  #Initial empty matrix
        
        #Coefficients
        MatFreq[0,0] = 1	
        MatFreq[0,1] = 0
        MatFreq[0,2] = 1
        MatFreq[0,3] = 0
        MatFreq[0,4] = 0
        MatFreq[0,5] = 0
        MatFreq[0,6] = 0
        MatFreq[0,7] = 0
        
        MatFreq[1,0] = 0
        MatFreq[1,1] = beta1
        MatFreq[1,2] = 0
        MatFreq[1,3] = beta1
        MatFreq[1,4] = 0
        MatFreq[1,5] = 0
        MatFreq[1,6] = 0
        MatFreq[1,7] = 0
        
        MatFreq[2,0] = np.cosh( geoDef['L1'] * beta1 )
        MatFreq[2,1] = np.sinh( geoDef['L1'] * beta1 )
        MatFreq[2,2] = np.cos( geoDef['L1'] * beta1 )
        MatFreq[2,3] = np.sin( geoDef['L1'] * beta1 )
        MatFreq[2,4] = -1 * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[2,5] = -1 * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[2,6] = -1 * np.cos( geoDef['L1'] * beta2 )
        MatFreq[2,7] = -1 * np.sin( geoDef['L1'] * beta2 )
        
        MatFreq[3,0] = beta1 * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[3,1] = beta1 * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[3,2] = -1 * beta1 * np.sin( geoDef['L1'] * beta1 )
        MatFreq[3,3] = beta1 * np.cos( geoDef['L1'] * beta1 )
        MatFreq[3,4] = -1 * beta2 * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[3,5] = -1 * beta2 * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[3,6] = beta2 * np.sin( geoDef['L1'] * beta2 )
        MatFreq[3,7] = -1 * beta2 * np.cos( geoDef['L1'] * beta2 )
           
        MatFreq[4,0] = geoDef['EI1'] * ( beta1 )**( 2 ) * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[4,1] = geoDef['EI1'] * ( beta1 )**( 2 ) * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[4,2] = -1 * geoDef['EI1'] * ( beta1 )**( 2 ) * np.cos( geoDef['L1'] * beta1 )
        MatFreq[4,3] = -1 * geoDef['EI1'] * ( beta1 )**( 2 ) * np.sin( geoDef['L1'] * beta1 )
        MatFreq[4,4] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[4,5] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[4,6] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.cos( geoDef['L1'] * beta2 )
        MatFreq[4,7] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.sin( geoDef['L1'] * beta2 )
        
        MatFreq[5,0] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[5,1] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[5,2] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.sin( geoDef['L1'] * beta1 )
        MatFreq[5,3] = -1 * geoDef['EI1'] * ( beta1 )**( 3 ) * np.cos( geoDef['L1'] * beta1 )
        MatFreq[5,4] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[5,5] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[5,6] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.sin( geoDef['L1'] * beta2 )
        MatFreq[5,7] = geoDef['EI2'] * ( beta2 )**( 3 ) * np.cos( geoDef['L1'] * beta2 )
        
        MatFreq[6,0] = 0
        MatFreq[6,1] = 0
        MatFreq[6,2] = 0
        MatFreq[6,3] = 0
        MatFreq[6,4] = ( beta2 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,5] = ( beta2 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,6] = -1 * ( beta2 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,7] = -1 * ( beta2 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
         
        MatFreq[7,0] = 0
        MatFreq[7,1] = 0
        MatFreq[7,2] = 0
        MatFreq[7,3] = 0
        MatFreq[7,4] = ( beta2 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,5] = ( beta2 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,6] = ( beta2 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,7] = -1 * ( beta2 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )   
            
        return MatFreq
    
    def SolveModes(self):
        self.FindNaturalFreq()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution']
                     
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation of the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)        
        
        self.xcor = [x1,x2]
              
        SolvedModes = []
        
        for idx,omega in enumerate(self.NaturalFreq):          
           
            beta1 = ( ( geoDef['EI1'] )**( -1 ) * geoDef['rhoA1'] * ( omega )**( 2 ) )**( 1/4 )
            beta2 = ( ( geoDef['EI2'] )**( -1 ) * geoDef['rhoA2'] * ( omega )**( 2 ) )**( 1/4 )
      
            C1,C2,C3,C4,C5,C6,C7,C8 = nf.SolveLinSys( self.MatrixFreq(omega) )
            
            shapeLen1 = lambda x: ( C3 * np.cos( x * beta1 ) + ( C1 * np.cosh( x * beta1 ) + ( C4 * np.sin( x * beta1 ) + C2 * np.sinh( x * beta1 ) ) ) )
            shapeLen2 = lambda x: ( C7 * np.cos( x * beta2 ) + ( C5 * np.cosh( x * beta2 ) + ( C8 * np.sin( x * beta2 ) + C6 * np.sinh( x * beta2 ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1)
            evalShapeLen2 = shapeLen2(x2)
            
            conOrt = integrate.trapz(geoDef['rhoA1']*evalShapeLen1**2,x1) + \
                     integrate.trapz(geoDef['rhoA2']*evalShapeLen2**2,x2)
                     
            mod1 = evalShapeLen1 * np.sqrt(1/conOrt)
            mod2 = evalShapeLen2 * np.sqrt(1/conOrt)
                        
            SolvedModes.append([mod1,mod2])
        
        self.SolvedModes = SolvedModes 
        
# =============================================================================
#  Cantilever Beam - 3 Sections
# =============================================================================

class CantileverBeam3(Modes):
    def __init__(self,input):
        Modes.__init__(self,input) #Inheritance

        hs = self.pars['hs']; bs = self.pars['bs']; Es = self.pars['Es']
        hp = self.pars['hp']; bp = self.pars['bp']; Ep = self.pars['Ep']
        L1 = self.pars['L1']; L2 = self.pars['L2']; L3 = self.pars['L3']
        
        rhos = self.pars['rhos']; rhop = self.pars['rhop'];
        
        e31 = self.pars['d31']*self.pars['Ep']; e311 = self.pars['e311']
        c111p = self.pars['c111p'] 
        
        self.geoDef = {
            'rhoA1': bs*hs*rhos,
            'rhoA2': ( bp*hp*rhop + bs*hs*rhos ),
            'rhoA3': bs*hs*rhos,
            'EI1': 1/12*bs*Es*(hs)**(3),
            'EI2': 1/12*((bp*Ep*hp + bs*Es*hs ))**(-1) * ((bp)**(2)*(Ep)**(2)*(hp)**(4) + ((bs)**(2)*(Es)**(2)*(hs)**(4) + 2*bp*bs*Ep*Es*hp*hs*(2*(hp)**(2) + (3*hp*hs + 2*(hs)**(2))))),
            'EI3': 1/12*bs*Es*(hs)**(3),
        
            'EA1': bs*hs*Es,
            'EA2': bs*hs*Es + bp*hp*Ep,
            'EA3': bs*hs*Es,
            
            'L1': L1, 'L2': L2, 'L3': L3,
            
            'EAn': bp * c111p * hp,
            'EBn': 1/2 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'EIn': 1/12 * bp * c111p * hp * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'EJn': 1/8 * bp * bs * c111p * Es * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -3 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 2 * ( hp )**( 2 ) + ( 2 * hp * hs + ( hs )**( 2 ) ) ) ) ),
            
            'theta0': bp * e31,
            'theta1': 1/2 * bp * bs * Es * e31 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'theta2': 1/12 * bp * e31 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),
            'psi0': bp * e311,
            'psi1': 1/2 * bp * bs * Es * e311 * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ),
            'psi2': 1/12 * bp * e311 * ( ( bp * Ep * hp + bs * Es * hs ) )**( -2 ) * ( ( bp )**( 2 ) * ( Ep )**( 2 ) * ( hp )**( 4 ) + ( 2 * bp * bs * Ep * Es * ( hp )**( 3 ) * hs + ( bs )**( 2 ) * ( Es )**( 2 ) * ( hs )**( 2 ) * ( 4 * ( hp )**( 2 ) + ( 6 * hp * hs + 3 * ( hs )**( 2 ) ) ) ) ),

            'chi': 1/2 * bp * bs * hp * hs * ( hp + hs ) * ( ( bp * Ep * hp + bs * Es * hs ) )**( -1 ) * ( -1 * Es * rhop + Ep * rhos )
            }
        
        self.SolveModes()
        self.SolveDerivates()
        self.PlotModes()
        
    def MatrixFreq(self,omega):
        geoDef = self.geoDef 
        
        beta1 = ( ( geoDef['EI1'] )**( -1 ) * geoDef['rhoA1'] * ( omega )**( 2 ) )**( 1/4 )
        beta2 = ( ( geoDef['EI2'] )**( -1 ) * geoDef['rhoA2'] * ( omega )**( 2 ) )**( 1/4 )
        beta3 = ( ( geoDef['EI3'] )**( -1 ) * geoDef['rhoA3'] * ( omega )**( 2 ) )**( 1/4 )

        MatFreq = np.zeros([12,12])  #Initial empty matrix
        
        MatFreq[0,0] = 1
        MatFreq[0,1] = 0
        MatFreq[0,2] = 1
        MatFreq[0,3] = 0
        MatFreq[0,4] = 0
        MatFreq[0,5] = 0
        MatFreq[0,6] = 0
        MatFreq[0,7] = 0
        MatFreq[0,8] = 0
        MatFreq[0,9] = 0
        MatFreq[0,10] = 0
        MatFreq[0,11] = 0
    
        MatFreq[1,0] = 0
        MatFreq[1,1] = beta1
        MatFreq[1,2] = 0
        MatFreq[1,3] = beta1
        MatFreq[1,4] = 0
        MatFreq[1,5] = 0
        MatFreq[1,6] = 0
        MatFreq[1,7] = 0
        MatFreq[1,8] = 0
        MatFreq[1,9] = 0
        MatFreq[1,10] = 0
        MatFreq[1,11] = 0

        MatFreq[2,0] = np.cosh( geoDef['L1'] * beta1 )
        MatFreq[2,1] = np.sinh( geoDef['L1'] * beta1 )
        MatFreq[2,2] = np.cos( geoDef['L1'] * beta1 )
        MatFreq[2,3] = np.sin( geoDef['L1'] * beta1 )
        MatFreq[2,4] = -1 * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[2,5] = -1 * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[2,6] = -1 * np.cos( geoDef['L1'] * beta2 )
        MatFreq[2,7] = -1 * np.sin( geoDef['L1'] * beta2 )
        MatFreq[2,8] = 0
        MatFreq[2,9] = 0
        MatFreq[2,10] = 0
        MatFreq[2,11] = 0

        MatFreq[3,0] = beta1 * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[3,1] = beta1 * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[3,2] = -1 * beta1 * np.sin( geoDef['L1'] * beta1 )
        MatFreq[3,3] = beta1 * np.cos( geoDef['L1'] * beta1 )
        MatFreq[3,4] = -1 * beta2 * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[3,5] = -1 * beta2 * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[3,6] = beta2 * np.sin( geoDef['L1'] * beta2 )
        MatFreq[3,7] = -1 * beta2 * np.cos( geoDef['L1'] * beta2 )
        MatFreq[3,8] = 0
        MatFreq[3,9] = 0
        MatFreq[3,10] = 0
        MatFreq[3,11] = 0

        MatFreq[4,0] = geoDef['EI1'] * ( beta1 )**( 2 ) * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[4,1] = geoDef['EI1'] * ( beta1 )**( 2 ) * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[4,2] = -1 * geoDef['EI1'] * ( beta1 )**( 2 ) * np.cos( geoDef['L1'] * beta1 )
        MatFreq[4,3] = -1 * geoDef['EI1'] * ( beta1 )**( 2 ) * np.sin( geoDef['L1'] * beta1 )
        MatFreq[4,4] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[4,5] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[4,6] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.cos( geoDef['L1'] * beta2 )
        MatFreq[4,7] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.sin( geoDef['L1'] * beta2 )
        MatFreq[4,8] = 0
        MatFreq[4,9] = 0
        MatFreq[4,10] = 0
        MatFreq[4,11] = 0

        MatFreq[5,0] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.sinh( geoDef['L1'] * beta1 )
        MatFreq[5,1] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.cosh( geoDef['L1'] * beta1 )
        MatFreq[5,2] = geoDef['EI1'] * ( beta1 )**( 3 ) * np.sin( geoDef['L1'] * beta1 )
        MatFreq[5,3] = -1 * geoDef['EI1'] * ( beta1 )**( 3 ) * np.cos( geoDef['L1'] * beta1 )
        MatFreq[5,4] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.sinh( geoDef['L1'] * beta2 )
        MatFreq[5,5] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.cosh( geoDef['L1'] * beta2 )
        MatFreq[5,6] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.sin( geoDef['L1'] * beta2 )
        MatFreq[5,7] = geoDef['EI2'] * ( beta2 )**( 3 ) * np.cos( geoDef['L1'] * beta2 )
        MatFreq[5,8] = 0
        MatFreq[5,9] = 0
        MatFreq[5,10] = 0
        MatFreq[5,11] = 0

        MatFreq[6,0] = 0
        MatFreq[6,1] = 0
        MatFreq[6,2] = 0
        MatFreq[6,3] = 0
        MatFreq[6,4] = np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,5] = np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,6] = np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,7] = np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[6,8] = -1 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[6,9] = -1 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[6,10] = -1 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[6,11] = -1 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )

        MatFreq[7,0] =  0
        MatFreq[7,1] = 0
        MatFreq[7,2] = 0
        MatFreq[7,3] = 0
        MatFreq[7,4] = beta2 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,5] = beta2 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,6] = -1 * beta2 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,7] = beta2 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[7,8] = -1 * beta3 * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[7,9] = -1 * beta3 * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[7,10] = beta3 * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[7,11] = -1 * beta3 * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        
        MatFreq[8,0] = 0
        MatFreq[8,1] = 0
        MatFreq[8,2] = 0
        MatFreq[8,3] = 0
        MatFreq[8,4] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[8,5] = geoDef['EI2'] * ( beta2 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[8,6] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[8,7] = -1 * geoDef['EI2'] * ( beta2 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[8,8] = -1 * geoDef['EI3'] * ( beta3 )**( 2 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[8,9] = -1 * geoDef['EI3'] * ( beta3 )**( 2 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[8,10] = geoDef['EI3'] * ( beta3 )**( 2 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[8,11] = geoDef['EI3'] * ( beta3 )**( 2 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )

        MatFreq[9,0] =  0
        MatFreq[9,1] = 0
        MatFreq[9,2] = 0
        MatFreq[9,3] = 0
        MatFreq[9,4] = geoDef['EI2'] * ( beta2 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[9,5] = geoDef['EI2'] * ( beta2 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[9,6] = geoDef['EI2'] * ( beta2 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[9,7] = -1 * geoDef['EI2'] * ( beta2 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta2 )
        MatFreq[9,8] = -1 * geoDef['EI3'] * ( beta3 )**( 3 ) * np.sinh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[9,9] = -1 * geoDef['EI3'] * ( beta3 )**( 3 ) * np.cosh( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[9,10] = -1 * geoDef['EI3'] * ( beta3 )**( 3 ) * np.sin( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )
        MatFreq[9,11] = geoDef['EI3'] * ( beta3 )**( 3 ) * np.cos( ( geoDef['L1'] + geoDef['L2'] ) * beta3 )

        MatFreq[10,0] = 0
        MatFreq[10,1] = 0
        MatFreq[10,2] = 0
        MatFreq[10,3] = 0
        MatFreq[10,4] = 0
        MatFreq[10,5] = 0
        MatFreq[10,6] = 0
        MatFreq[10,7] = 0
        MatFreq[10,8] = ( beta3 )**( 2 ) * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[10,9] = ( beta3 )**( 2 ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[10,10] = -1 * ( beta3 )**( 2 ) * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[10,11] = -1 * ( beta3 )**( 2 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
      
        MatFreq[11,0] = 0
        MatFreq[11,1] = 0
        MatFreq[11,2] = 0
        MatFreq[11,3] = 0
        MatFreq[11,4] = 0
        MatFreq[11,5] = 0
        MatFreq[11,6] = 0
        MatFreq[11,7] = 0
        MatFreq[11,8] = ( beta3 )**( 3 ) * np.sinh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[11,9] = ( beta3 )**( 3 ) * np.cosh( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[11,10] = ( beta3 )**( 3 ) * np.sin( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )
        MatFreq[11,11] = -1 * ( beta3 )**( 3 ) * np.cos( ( geoDef['L1'] + ( geoDef['L2'] + geoDef['L3'] ) ) * beta3 )

        return MatFreq

    def SolveModes(self):
        
        self.FindNaturalFreq()
        
        geoDef = self.geoDef
        modeRes = self.modesConf['modeResolution']
                        
        x1 = np.linspace(0, geoDef['L1'], num = modeRes) #Discreate evaluation of the modes
        x2 = np.linspace(geoDef['L1'], geoDef['L1'] + geoDef['L2'], num = modeRes)
        x3 = np.linspace(geoDef['L1'] + geoDef['L2'] ,geoDef['L1'] + geoDef['L2'] + geoDef['L3'] , num = modeRes)
        
        self.xcor = [x1,x2,x3]
               
        SolvedModes = []
        
        for idx,omega in enumerate(self.NaturalFreq):          
            beta1 = (( geoDef['EI1'] )**(-1) * geoDef['rhoA1'] * ( omega )**(2) )**(1/4)
            beta2 = (( geoDef['EI2'] )**(-1) * geoDef['rhoA2'] * ( omega )**(2) )**(1/4)
            beta3 = (( geoDef['EI3'] )**(-1) * geoDef['rhoA3'] * ( omega )**(2) )**(1/4)
        
            C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12 = nf.SolveLinSys( self.MatrixFreq(omega) )

            shapeLen1 = lambda x: ( C3 * np.cos( x * beta1 ) + ( C1 * np.cosh( x * beta1 ) + ( C4 * np.sin( x * beta1 ) + C2 * np.sinh( x * beta1 ) ) ) )
            shapeLen2 = lambda x: ( C7 * np.cos( x * beta2 ) + ( C5 * np.cosh( x * beta2 ) + ( C8 * np.sin( x * beta2 ) + C6 * np.sinh( x * beta2 ) ) ) )         
            shapeLen3 = lambda x: ( C11 * np.cos( x * beta3 ) + ( C9 * np.cosh( x * beta3 ) + ( C12 * np.sin( x * beta3 ) + C10 * np.sinh( x * beta3 ) ) ) )
            
            evalShapeLen1 = shapeLen1(x1)
            evalShapeLen2 = shapeLen2(x2)
            evalShapeLen3 = shapeLen3(x3)
            
            conOrt = integrate.trapz(geoDef['rhoA1']*evalShapeLen1**2,x1) + \
                     integrate.trapz(geoDef['rhoA2']*evalShapeLen2**2,x2) + \
                     integrate.trapz(geoDef['rhoA3']*evalShapeLen3**2,x3)
                     
            mod1 = evalShapeLen1 * np.sqrt(1/conOrt)
            mod2 = evalShapeLen2 * np.sqrt(1/conOrt)
            mod3 = evalShapeLen3 * np.sqrt(1/conOrt)
                        
            SolvedModes.append([mod1,mod2,mod3])
        
        self.SolvedModes = SolvedModes 

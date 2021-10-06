import sys
from solvers import solMMS
from solvers import solNUM 
from solvers import solEXP
from auxiliar import plots as pp
import numpy as np
import pprint
import copy

    
# =============================================================================
# Frecuency response solutions
# =============================================================================

class solution(object):
    def __repr__(self): return 'Solution_element'
    def __init__(self,input,system):
        # Importing scalar values
        self.Rl = system.Rl
        self.Cp = system.Cp
        
        # Importing linear matrix 
        self.FF = system.FF
        self.MM = system.MM
        self.KK = system.KK
        self.CC = system.CC
        self.theta = system.theta
        self.psi = system.psi
        
        # Importing nonlinear matrix
        self.MMM = system.MMM
        self.CCC = system.CCC
        self.Kg = system.Kg #Nonlinear Stiffness
        self.thetaG = system.thetaG
        self.psiG = system.psiG
        self.thetaC = system.thetaC
        self.psiC = system.psiC
        self.Kc = system.Kc
        
        #Nonlinear parameters from postbuckling conf
        self.Kgq = system.Kgq
        self.Kcq = system.Kcq
        self.thetaGq = system.thetaGq
        self.thetaCq = system.thetaCq
        self.psiGq = system.psiGq
        self.psiCq = system.psiCq
        
        # Usefull data
        self.nModes = system.nModes
        self.acelValues = input['acelValues']
        self.solRanges = input['solRanges']
        self.expName = input['expSample']
        self.setup = input['setup']
        
        # Mode evaluated at specific position
        self.modeEval = system.modeEval
        
        # Running solution for MMS
        if input['setup']['solve_MMS'] == 'y':
            self.solveMMS()
        elif input['setup']['solve_MMS'] == 'n':
            pass
        else:
            print("solve_MMS must be 'y' or 'n'")
            sys.exit()
         
        # Running solution for Num Integration
        if input['setup']['solve_Num'] == 'y':
            self.solveNUM()
        elif input['setup']['solve_Num'] == 'n':
            pass
        else:
            print("solve_Num must be 'y' or 'n'")
            sys.exit()
                
        # Running solution for Experimental Solution
        if input['setup']['solve_Exp'] == 'y':
            self.solveEXP()
        elif input['setup']['solve_Exp'] == 'n':
            pass
        else:
            print("solve_Exp must be 'y' or 'n'")
            sys.exit()
        
        # Plots
        self.plotFRF()

# =============================================================================
# MMS solution
# =============================================================================
    def solveMMS(self):
        #No modal interaction is considerated
        fiMSM = self.solRanges['fiMSM']; ffMSM = self.solRanges['ffMSM']
        minMSM = self.solRanges['minMSM']; maxMSM = self.solRanges['maxMSM']
        nMSM  = self.solRanges['nMSM']

        self.solMMS = {}
        
        for acc in self.acelValues:   
            text = "Solving MMS for acc= "+str(acc)+"g"; print(text) 
            for j in range(self.nModes):
                amp = np.linspace(minMSM,maxMSM, int(nMSM)) #Expected amplitud values for disp
                
                parsLoop = {
                        'FF' : self.FF[j]*acc,
                        'MM': self.MM[j,j],
                        'MMM': self.MMM[j,j,j],
                        'CC': self.CC[j,j],
                        'CCC': self.CCC[j,j,j],
                        'Kc': self.Kc[j,j,j],
                        'Kg': self.Kg[j,j,j,j],
                        'theta': self.theta[j],
                        'thetaC': self.thetaC[j,j],
                        'thetaG': self.thetaG[j,j],
                        'psi' : self.psi[j],
                        'psiG' : self.psiG[j,j],
                        'psiC' : self.psiC[j,j],
                        'wn': (self.KK[j,j])**0.5,
                        
                        'Kcq': self.Kcq[j,j],
                        'Kgq': self.Kgq[j,j,j],
                        'thetaCq': self.thetaCq[j],
                        'thetaGq': self.thetaGq[j],
                        'psiCq' : self.psiCq[j],
                        'psiGq' : self.psiGq[j],

                        'Rl':self.Rl,
                        'Cp':self.Cp }
                
                freq1, freq2, volt1, volt2, gamma = solMMS.MultipleScales(amp,parsLoop)
                
                # cutting the solutions where are expected
                x1 = np.where( (freq1>fiMSM) & (freq1<ffMSM)); x2 = np.where( (freq2>fiMSM) & (freq2<ffMSM))      
                freq1 = freq1[x1];      freq2 = freq2[x2]; 
                amp1 = amp[x1];         amp2 = amp[x2]; 
                volt1 = volt1[x1];      volt2 = volt2[x2]; 
                gamma1 = gamma[x1];      gamma2=gamma[x2];
                
                #inverting the second solution
                freq2 = freq2[::-1]; amp2 = amp2[::-1] ; volt2 = volt2[::-1]
                freq =  np.hstack((freq1,freq2))
                amp =   np.hstack((amp1,amp2))
                volt =  np.hstack((volt1,volt2))
                gamma = np.hstack((gamma1,gamma2))
                
                # =============================================================================
                #                 Test for maximuym voltage (adding harmonicss)
                # =============================================================================
                voltMax = np.zeros( np.shape(volt) )
                freqMax = copy.deepcopy(freq)
                
                for idx,ff in enumerate(freqMax):
                    wf = 2*np.pi*ff
 
                    #Harmonic parameters
                    R = self.Rl; wn = (self.KK[j,j])**0.5; Cp =self.Cp; 
                    psi = self.psi[j]
                    psiG = self.psiG[j,j]
                    psiC = self.psiC[j,j]
                    psiGq = self.psiGq[j]
                    psiCq = self.psiCq[j]
                    alpha2 = self.Kgq[j,j,j]
                    chi = self.MMM[j,j,j]
                    
                    #Harmonic amplitudes
                    K1 = 1/3 * ( np.pi )**( -1 ) * ( ( amp[idx] )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( 16 * ( amp[idx] )**( 2 ) * ( psiC )**( 2 ) + ( 24 * ( ( amp[idx] )**( 2 ) )**( 1/2 ) * np.pi * psiC * ( psi + psiGq ) + 9 * ( np.pi )**( 2 ) * ( ( psi + psiGq ) )**( 2 ) ) ) )**( 1/2 )
                    K2 = 1/6 * ( np.pi )**( -1 ) * ( ( amp[idx] )**( 2 ) * ( R )**( 2 ) * ( ( ( wn )**( 2 ) + 4 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 4 ) ) )**( -1 ) * ( ( amp[idx] )**( 2 ) * ( np.pi )**( 2 ) * ( ( 2 * alpha2 * ( psi + psiGq ) + ( wn )**( 2 ) * ( 3 * psiG + -4 * chi * ( psi + psiGq ) ) ) )**( 2 ) + 32 * ( wn )**( 2 ) * psiCq * ( 8 * ( wn )**( 2 ) * psiCq + ( ( amp[idx] )**( 2 ) )**( 1/2 ) * np.pi * ( 2 * alpha2 * ( psi + psiGq ) + ( wn )**( 2 ) * ( 3 * psiG + -4 * chi * ( psi + psiGq ) ) ) ) ) )**( 1/2 )
                    K3 = 4/5 * ( np.pi )**( -1 ) * ( ( amp[idx] )**( 4 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 9 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiC )**( 2 ) )**( 1/2 )
                    K4 = 16/15 * ( np.pi )**( -1 ) * ( ( amp[idx] )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 16 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiCq )**( 2 ) )**( 1/2 )
                    K5 = 4/21 * ( np.pi )**( -1 ) * ( ( amp[idx] )**( 4 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 25 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiC )**( 2 ) )**( 1/2 )
                               
                    #Phase angles
                    theta1 = -1 * np.arctan( Cp * R * wn )
                    theta2 = -1 * np.arctan( 2 * Cp * R * wn )
                    theta3 = -1 * np.arctan( 3 * Cp * R * wn )
                    theta4 = -1 * np.arctan( 4 * Cp * R * wn )
                    theta5 = -1 * np.arctan( 5 * Cp * R * wn )
                    
                    #Time lapse for 1 complete cycles
                    time_lapse = np.linspace(0, 1/ff, num = 500)
                    
                    #Harmonic functions
                    firstHarFun = lambda t: K1*np.cos(wf*t + gamma[idx] + theta1)
                    seconHarFun = lambda t: K2*np.cos(2*wf*t + 2*gamma[idx] + theta2)
                    thirdHarFun = lambda t:  K3*np.cos(3*wf*t + 3*gamma[idx] + theta3)
                    fourthHarFun = lambda t: K4*np.cos(4*wf*t + 4*gamma[idx] + theta4)
                    fifthHarFun = lambda t: K5*np.cos(5*wf*t + 5*gamma[idx] + theta5)
                   
                    #Evaluation of the harmonic functions
                    firstHar = firstHarFun(time_lapse)
                    seconHar = seconHarFun(time_lapse)
                    thirdHar = thirdHarFun(time_lapse)
                    fourthHar = fourthHarFun(time_lapse)
                    fifthHar = fifthHarFun(time_lapse)

                    vol = firstHar + seconHar + thirdHar + fourthHar + fifthHar
 
                    voltMax[idx] = max( abs(vol) )
                    
                # =============================================================================
                #               End
                # =============================================================================
                #multiplying the generalized displacement by each evaluated mode
                amp = amp * abs( self.modeEval[j] )

                nameSol = str(acc)+"g, Mod#" +str(j+1)

                self.solMMS[nameSol] = (freq, amp ,voltMax)

    def solveMMSTemp(self, freq , acc = 1, mod = 0, plot = False):
        #No modal interaction is considerated
        minMSM = self.solRanges['minMSM']; maxMSM = self.solRanges['maxMSM']
        nMSM  = self.solRanges['nMSM']*2

        amp = np.linspace(minMSM,maxMSM, int(nMSM)) #Expected amplitud values for disp
                
        pars = {
                'FF' : self.FF[mod]*acc,
                'MM': self.MM[mod,mod],
                'MMM': self.MMM[mod,mod],
                'CC': self.CC[mod,mod],
                'CCC': self.CCC[mod,mod,mod],
                'Kc': self.Kc[mod,mod,mod],
                'Kg': self.Kg[mod,mod,mod,mod],
                'theta': self.theta[mod],
                'thetaC': self.thetaC[mod,mod],
                'thetaG': self.thetaG[mod,mod],
                'psi' : self.psi[mod],
                'psiG' : self.psiG[mod,mod],
                'psiC' : self.psiC[mod,mod],
                'wn': (self.KK[mod,mod])**0.5,
                
                'Kcq': self.Kcq[mod,mod],
                'Kgq': self.Kgq[mod,mod,mod],
                'thetaCq': self.thetaCq[mod],
                'thetaGq': self.thetaGq[mod],
                'psiCq' : self.psiCq[mod],
                'psiGq' : self.psiGq[mod],

                'Rl':self.Rl,
                'Cp':self.Cp }
                
        #Solving the complete curve
        freq1, freq2, volt1, volt2, gamma = solMMS.MultipleScales(amp, pars)
        
        #Looking for the specifided frequency in the complete curve
        epsilon = 0.05
        try:
            fpos = np.where( (freq2 - epsilon <= freq) & (freq2 + epsilon >= freq))[0][0]
            fnew = freq2[fpos]
            gammaNew = gamma[fpos]
        except:
            fpos = np.where( (freq1 - epsilon <= freq) & (freq1 + epsilon >= freq))[0][0]
            fnew = freq1[fpos]
            gammaNew = gamma[fpos]
        
        a = amp[fpos]
        wf = 2*np.pi*fnew
        print('Be carefull, with unestable or highest energy solutions for MMS-Temp')
        print("Frequency findend: "+str(round(fnew,2))+"Hz")

        #Harmonic parameters
        R = self.Rl; wn = (self.KK[mod,mod])**0.5; Cp =self.Cp; 
        psi = self.psi[mod]
        psiG = self.psiG[mod,mod]
        psiC = self.psiC[mod,mod]
        psiGq = self.psiGq[mod]
        psiCq = self.psiCq[mod]
        alpha2 = self.Kgq[mod,mod,mod]
        chi = self.MMM[mod,mod,mod]
        
        #Harmonic amplitudes
        K1 = 1/3 * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( 16 * ( a )**( 2 ) * ( psiC )**( 2 ) + ( 24 * ( ( a )**( 2 ) )**( 1/2 ) * np.pi * psiC * ( psi + psiGq ) + 9 * ( np.pi )**( 2 ) * ( ( psi + psiGq ) )**( 2 ) ) ) )**( 1/2 )
        K2 = 1/6 * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( R )**( 2 ) * ( ( ( wn )**( 2 ) + 4 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 4 ) ) )**( -1 ) * ( ( a )**( 2 ) * ( np.pi )**( 2 ) * ( ( 2 * alpha2 * ( psi + psiGq ) + ( wn )**( 2 ) * ( 3 * psiG + -4 * chi * ( psi + psiGq ) ) ) )**( 2 ) + 32 * ( wn )**( 2 ) * psiCq * ( 8 * ( wn )**( 2 ) * psiCq + ( ( a )**( 2 ) )**( 1/2 ) * np.pi * ( 2 * alpha2 * ( psi + psiGq ) + ( wn )**( 2 ) * ( 3 * psiG + -4 * chi * ( psi + psiGq ) ) ) ) ) )**( 1/2 )
        K3 = 4/5 * ( np.pi )**( -1 ) * ( ( a )**( 4 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 9 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiC )**( 2 ) )**( 1/2 )
        K4 = 16/15 * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 16 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiCq )**( 2 ) )**( 1/2 )
        K5 = 4/21 * ( np.pi )**( -1 ) * ( ( a )**( 4 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + 25 * ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( psiC )**( 2 ) )**( 1/2 )

        #Phase angles
        theta1 = -1 * np.arctan( Cp * R * wn )
        theta2 = -1 * np.arctan( 2 * Cp * R * wn )
        theta3 = -1 * np.arctan( 3 * Cp * R * wn )
        theta4 = -1 * np.arctan( 4 * Cp * R * wn )
        theta5 = -1 * np.arctan( 5 * Cp * R * wn )
          
        #Time lapse for 5 complete cycles
        time_lapse = np.linspace(0, 5/fnew, num = 500)
        
        #Harmonic functions
        dispFun = lambda t: a*np.cos(wf*t)
        firstHarFun = lambda t: K1*np.cos(wf*t + gammaNew + theta1)
        seconHarFun = lambda t: K2*np.cos(2*wf*t + 2*gammaNew + theta2)
        thirdHarFun = lambda t:  K3*np.cos(3*wf*t + 3*gammaNew + theta3)
        fourthHarFun = lambda t: K4*np.cos(4*wf*t + 4*gammaNew + theta4)
        fifthHarFun = lambda t: K5*np.cos(5*wf*t + 5*gammaNew + theta5)
       
        #Evaluation of the harmonic functions
        disp = dispFun(time_lapse)* abs( self.modeEval[mod] )
        firstHar = firstHarFun(time_lapse)
        seconHar = seconHarFun(time_lapse)
        thirdHar = thirdHarFun(time_lapse)
        fourthHar = fourthHarFun(time_lapse)
        fifthHar = fifthHarFun(time_lapse)
        
        vol = firstHar + seconHar + thirdHar + fourthHar + fifthHar
        
        self.solMMSTemp = (time_lapse, disp, firstHar, seconHar, thirdHar,fourthHar,fifthHar, vol)
        
        if plot is True:
            pp.plotMMSTemp(self, freq, acc)
        elif plot is False:
            pass
        else: 
            print("Plot option must be a boolean")
            sys.exit()
        
# =============================================================================
# Numerical Integration
# =============================================================================

    def solveNUM(self):
        # Importing range solutions
        fi1 = self.solRanges['fi1']; ff1 = self.solRanges['ff1']; n1 = self.solRanges['n1']
        fi2 = self.solRanges['fi2']; ff2 = self.solRanges['ff2']; n2 = self.solRanges['n2']
        fi3 = self.solRanges['fi3']; ff3 = self.solRanges['ff3']; n3 = self.solRanges['n3']
        
        # Frequency expected values
        sw1 = np.linspace(fi1,ff1,n1); sw2 = np.linspace(fi2,ff2,n2); sw3 = np.linspace(fi3,ff3,n3)
        freq = np.concatenate((sw1,sw2,sw3))
        
        self.solNUM = {}
        
        for acc in self.acelValues:
            # Empty solution lists
            ampPeak = np.zeros(len(freq)); velPeak = np.zeros(len(freq)); volPeak = np.zeros(len(freq))
            
            for idx,f in enumerate(freq): #Barrido en frecuencia
                text = "("+str(idx+1)+"/"+str(len(freq))+")"; print(text)
                
                print('Solving NUM for f='+str(round(f,1))+' and a='+str(acc))
                solA, solVel, solVol = solNUM.NumSystemPeak(f,self,acc)
                
                for i in range(self.nModes):
                    ampPeak[idx] += abs(solA[i]* self.modeEval[i])
                    velPeak[idx] += abs(solVel[i]*self.modeEval[i])
                
                volPeak[idx] = abs(solVol)
                
            nameSol = str(acc)+"g"
            self.solNUM[nameSol] = (freq, ampPeak , volPeak)

    def solveNUMTemp(self, freq, acc = 1,plot = True):
        self.solNUMTemp = solNUM.NumSystemSol(freq,self,acc)
        
        for i in range(self.nModes):
            self.solNUMTemp.y[i] *=  self.modeEval[i]
        
        if plot is True:
            pp.plotNUMTemp(self, freq, acc)
        elif plot is False:
            pass
        else: 
            print("plot option must be a boolean")
            sys.exit()
        
# ============================================================================
#  Experimental
# ============================================================================

    def solveEXP(self):
        if self.setup['solve_disp']=='y': #empty dict for disp
                    self.solEXPdisp = {}
        elif self.setup['solve_disp']=='n':
            pass
        else:
            print("'solve_disp' must be 'y' or 'n'")
            sys.exit()
            
        if self.setup['solve_volt']=='y': #empty dict for volt
            self.solEXPvolt = {}
        elif self.setup['solve_volt']=='n':
            pass
        else:
            print("'solve_volt must be 'y' or 'n'")
            sys.exit()
        
        for i in self.expName:
            for j in self.acelValues:
                val = str(j)
                valAcc = val.replace(".",",")
                
                val = str( int(self.Rl*1E-3) )
                valRl = val.replace(".",",")
                
                nameExp = i + "a" + valAcc + "r" + valRl + ".txt"
                
                if self.setup['solve_disp']=='y': 
                    try:
                        self.solEXPdisp[nameExp] = solEXP.importData('experimental/displacement/' + nameExp)
                    except:
                        print(nameExp + ' from disp not found')
                        
                if self.setup['solve_volt']=='y': 
                    try:
                        self.solEXPvolt[nameExp] = solEXP.importData('experimental/generation/' + nameExp)
                    except:
                        print(nameExp + ' from volt not found')            

# =============================================================================
# Others
# =============================================================================

    def plotFRF(self):
        pp.plotFRF(self)

    def plotCompMMSandNUM(self):
        pp.plotCompMMSandNUM(self)     
                
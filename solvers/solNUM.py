import numpy as np
from scipy.integrate import solve_ivp

def NumSystem(t,X,wf,solEl,acc): #X is a three elements size vector

    MM = solEl.MM
    FF = solEl.FF * np.cos(wf*t) *acc
    CC = solEl.CC
    KK = solEl.KK
    CCC = solEl.CCC
    Kg = solEl.Kg
    Kc = solEl.Kc
    theta = solEl.theta
    thetaC = solEl.thetaC
    thetaG = solEl.thetaG
    psi = solEl.psi
    psiC = solEl.psiC
    psiG = solEl.psiG
    Rl = solEl.Rl
    Cp = solEl.Cp   
    
    MMM = solEl.MMM
    Kgq = solEl.Kgq
    Kcq = solEl.Kcq
    thetaGq = solEl.thetaGq
    thetaCq = solEl.thetaCq
    psiGq = solEl.psiGq
    psiCq = solEl.psiCq
    
    nModes = solEl.nModes   

    XP = np.zeros ([2*nModes + 1]) # +1 for the electric equation
    
    XP[ (2*nModes) ] = -X[(2*nModes)]*(Rl*Cp)**(-1) #electric equation
    
    for i in range(nModes):
        XP[i] = X[i+nModes]
        XP[i + nModes] = MM[i,i]**-1 * ( FF[i] - CC[i,i]*X[i+nModes] - KK[i,i]*X[i] - theta[i]*X[(2*nModes)] - thetaGq[i]*X[(2*nModes)] - thetaCq[i]*X[(2*nModes)] * np.sign(X[i]) )
        XP[ (2*nModes) ] += -X[i+nModes] * psi[i] * (1/Cp) -X[i+nModes] * psiGq[i] * (1/Cp) - X[i+nModes] * np.sign(X[i]) * psiCq[i] * (1/Cp) 
        
        for j in range(nModes):
            XP[i + nModes] +=  -thetaG[i,j] * X[(2*nModes)] * X[j] - thetaC[i,j] * X[(2*nModes)] * X[j] *np.sign(X[j]) - Kcq[i,j]*X[j]*np.sign(X[j])
            XP[ (2*nModes) ] += -psiG[i,j] * X[i+nModes] * X[j] * (1/Cp) - psiC[i,j] * X[j+nModes] * np.abs(X[j]) * (1/Cp)
            
            for k in range(nModes):
                XP[i + nModes] +=  -MMM[i,j,k] * X[j+nModes] * X[k+nModes] - CCC[i,j,k] * X[j+nModes] * X[k+nModes] * np.sign(X[j+nModes]) - Kc[i,j,k]*X[j]*X[k]*np.sign(X[j]) - Kgq[i,j,k]*X[j]*X[k] 
                XP[ (2*nModes) ] += 0
                
                for l in range(nModes):
                    XP[i + nModes] +=  - Kg[i,j,k,l]*X[j]*X[k]*X[l] 
    
    return XP

def NumSystemSol(f,solEl,acc):
    wf = 2*np.pi*f #forcing frequency
    
    tinit = 0; tend = 10; nsample = 50
    
    t0 = [tinit,tend] #time span
    sample = np.linspace(tinit,tend, int( f*tend*nsample ))
    
    X0 = np.zeros([ 2*solEl.nModes + 1]) #initial conditions
    
    fun = lambda t,X : NumSystem(t,X,wf,solEl,acc)
    sol = solve_ivp(fun, t0 ,X0,  method= 'BDF', t_eval = sample)#Solvers RK45, RK23, LSODA, BDF

    return sol

def NumSystemPeak(f,solEl,acc):
    sol = NumSystemSol(f,solEl,acc).y
    nModes = solEl.nModes
    
    peaks = np.zeros(nModes*2 + 1)

    for idx,i in enumerate(sol):
        p1 = int(0.8*len(i)); p2 = int(len(i)-1)
        peaks[idx] = 0.5*( max(i[p1:p2]) - min(i[p1:p2]))
        
    peakPos = peaks[:nModes]
    peakVel = peaks[nModes:2*nModes]
    peakVol = peaks[-1]

    return peakPos, peakVel, peakVol
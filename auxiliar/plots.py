import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import tikzplotlib

save_plots = False
namePlot = 'test-'

# Font
plt.rcParams['font.family'] = 'serif'

#Interpolation experimental curves
nIntEXP = 1000 #Number interpolation from the experimental results
SVFwindow = 101 #Savitzky–Golay filter window lenght
SVFpolOrd = 3 #Savitzky–Golay polyorder

def plotFreqEquation(wVal,detVal):
    plt.figure()
    plt.title('Finding roots from Frequency Equation')
    plt.plot( wVal, detVal,'-bo')
    plt.yscale('log')
    plt.xlabel('Circular Frequency [rad/s]')
    plt.ylabel('Determinant of the matrix')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "FreqEq", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "FreqEq", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "FreqEq.tex")
    
def plotBuckEquation(PVal,detVal):    
    plt.figure()
    plt.title('Finding roots from Buckling Equation')
    plt.plot( PVal, detVal,'-ro')
    plt.yscale('log')
    plt.xlabel('Axial Load [N]')
    plt.ylabel('Determinant of the matrix')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "BuckEq", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "BuckEq", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "BuckEq.tex")

def plotBuckModes(modesEl):
    
    """
    modesEl is a mode-shape element
    """
    
    xcor = modesEl.xcorbuck; SolvedModes = modesEl.SolvedBuckModes
    numSections = len(xcor); numModes = len(SolvedModes)
    
    colors = plt.cm.rainbow(np.linspace(0,1,numModes))
    
    plt.figure()
    plt.title('Buckled modes with real displacement')
    for idx,mode in enumerate(SolvedModes):
        for sec in range(numSections):
            if sec is 0:
                lab = "Mode 0"+str(idx+1)
                plt.plot(xcor[sec],mode[sec],label = lab, c=colors[idx])
            else:
                plt.plot(xcor[sec],mode[sec],c=colors[idx])
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.xlabel('x-cordinate [m]')
    plt.ylabel('Amplitude [m]')
    plt.tight_layout()
    plt.show()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "BuckModes", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "BuckModes", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "BuckModes.tex")

def plotModes(modesEl):
    
    """
    modesEl is a mode-shape element
    """
    
    xcor = modesEl.xcor; SolvedModes = modesEl.SolvedModes
    numSections = len(xcor); numModes = len(SolvedModes)
    
    colors = plt.cm.rainbow(np.linspace(0,1,numModes))
    
    plt.figure()
    plt.title('Normalized mode shapes')
    for idx,mode in enumerate(SolvedModes):
        for sec in range(numSections):
            if sec is 0:
                lab = "Mode 0"+str(idx+1)
                plt.plot(xcor[sec],mode[sec],label = lab, c=colors[idx])
            else:
                plt.plot(xcor[sec],mode[sec],c=colors[idx])
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.xlabel('x-cordinate [m]')
    plt.ylabel('Amplitude [m]')
    plt.tight_layout()
    plt.show()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "Modes", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "Modes", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "Modes.tex")
    

def plotFRF(solEl):
    #solEL is a solution element which constant the three solutions
    
# =============================================================================
# # Voltage graph
# =============================================================================
    if solEl.setup['solve_volt']=='y': #empty dict for disp
        plt.figure()
        plt.title('Generation')
        
        # MMS solution
        try:
            colors = plt.cm.Set1(np.linspace(0,1,len(solEl.solMMS)))
            idx = 0
            for ref,data in solEl.solMMS.items():
                plt.plot( data[0], data[2], linewidth=2, label = ref, c = colors[idx])
                idx+= 1
        except: 
            pass
        
        # Experimental solution
        try:
            colors = plt.cm.Set1(np.linspace(0,1,len(solEl.solEXPvolt)))
            idx = 0
            for ref,data in solEl.solEXPvolt.items():
                x = data[0]; y = data[1]
                fint = interp1d(x, y, kind = 'linear') 
                xnew = np.linspace(min(x), max(x), num = nIntEXP, endpoint = True)
                ynew = fint(xnew); ynew = savgol_filter(ynew, SVFwindow, SVFpolOrd)    
                if idx == 0:
                    plt.plot( x, y, 'o', c = colors[idx],markersize=4)
                    # plt.plot( xnew, ynew, '-', c = colors[idx])
                else:
                    plt.plot( x, y, 'o', c = colors[idx],markersize=4)
                    # plt.plot( xnew, ynew, '-',c = colors[idx])
                idx += 1
        except:
            pass
        
        # NUM solution
        try:
            idx = 0
            for ref,data in solEl.solNUM.items():
                plt.plot( data[0], data[2], 'o', label = ref, c = colors[idx])
                idx+=1
        except: 
            pass
        
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Generation[V]')
        plt.legend(loc = 'upper right')
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        if save_plots is True:
            plt.savefig("output/png/" + namePlot + "Volt", format = 'png')
            plt.savefig("output/pdf/" + namePlot + "Volt", format = 'pdf')
            tikzplotlib.save("output/tex/" + namePlot + "Volt.tex")
    
    elif solEl.setup['solve_volt']=='n':
        pass
    else:
        print("'solve_volt' must be 'y' or 'n'")
        sys.exit()
        
# =============================================================================
# # Disp graph
# =============================================================================
    if solEl.setup['solve_disp']=='y': #empty dict for disp
        plt.figure()
        plt.title('Disp graph')
        
        # MMS solution
        try:
            colors = plt.cm.Set1(np.linspace(0,1,len(solEl.solMMS)))
            idx = 0
            for ref,data in solEl.solMMS.items():
                plt.plot( data[0], data[1], linewidth=2, label = ref, c = colors[idx])
                idx+=1
        except: 
            pass
        
        # Experimental solution
        try:
            idx = 0
            for ref,data in solEl.solEXPdisp.items():
                x = data[0]; y = data[1]  
                if idx == 0:
                    plt.plot( x, y, 'bo-',label = "Experiment", markersize=1.5)
                else:
                    plt.plot( x, y, 'bo-')
                idx+=1
        except:
            pass
        
        # NUM solution
        try:
            colors = plt.cm.Set1(np.linspace(0,1,len(solEl.solNUM)))
            idx = 0
            for ref,data in solEl.solNUM.items():
                plt.plot( data[0], data[1], 'o', label = ref, c = colors[idx])
                idx+=1
        except: 
            pass
            
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Displacement')
        plt.legend(loc = 'upper right')
        plt.grid()
        plt.tight_layout()
        plt.show()
        
        if save_plots is True:
            plt.savefig("output/png/" + namePlot + "Disp", format = 'png')
            plt.savefig("output/pdf/" + namePlot + "Disp", format = 'pdf')
            tikzplotlib.save("output/tex/" + namePlot + "Disp.tex")

    elif solEl.setup['solve_disp']=='n':
        pass
    else:
        print("'solve_disp' must be 'y' or 'n'")
        sys.exit()

def plotMMSTemp(solEl, freq ,acc):
    #solEL is a solution element which constant the three solutions
    data = solEl.solMMSTemp
    
    plt.figure()
    plt.suptitle('Temporal generation and displacement at vibrometer position for '+str(freq)+'Hz and '+str(acc)+'g')

    plt.subplot(2,1,1)
    plt.plot(data[0], data[1],'b-',markersize = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(data[0], data[2],'r-',linewidth = 1,markersize = 1, label = "First harmonic")
    plt.plot(data[0], data[3],'r-',linewidth = 1,markersize = 1, label = "Second harmonic")
    plt.plot(data[0], data[4],'r-',linewidth = 1,markersize = 1, label = "Third harmonic")
    plt.plot(data[0], data[5],'r-',linewidth = 1,markersize = 1, label = "Fourth harmonic")
    plt.plot(data[0], data[6],'r-',linewidth = 1,markersize = 1, label = "Fifth harmonic")
    plt.plot(data[0], data[7],'b-',linewidth = 2,markersize = 1, label = "Generation")
    plt.xlabel('Time [s]')
    plt.ylabel('Generation [V]')
    plt.grid()
    plt.tight_layout()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "MMSTemp", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "MMSTemp", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "MMSTemp.tex")
    
def plotNUMTemp(solEl, freq ,acc):
    #solEL is a solution element which constant the three solutions
    data = solEl.solNUMTemp.y
    time = solEl.solNUMTemp.t
    nModes = solEl.nModes
    
    plt.figure()
    plt.suptitle('Temporal generation and displacement at vibrometer position for '+str(freq)+'Hz and '+str(acc)+'g')

    plt.subplot(3,1,1)
    for i in range(nModes):
        plt.plot(time, data[i],'o-',markersize = 1, label = "Mod#" + str(i+1))
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.legend(loc = "upper right")
    plt.grid()
    
    plt.subplot(3,1,2)
    for i in range(nModes):
        plt.plot(time, data[i+nModes],'o-',markersize = 1, label = "Mod#" + str(i+1))
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(time, data[-1],'o-',markersize = 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Generation [V]')
    plt.grid()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "NUMTemp", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "NUMTemp", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "NUMTemp.tex")
    
def plotCompMMSandNUM(solEl):
    dataMMS = solEl.solMMSTemp
    dataNUM = solEl.solNUMTemp
            
    #dispMMS
    idxInit = 1
    while np.sign(dataMMS[1][idxInit] ) == np.sign(dataMMS[1][idxInit + 1] ) \
        or np.sign(dataMMS[1][idxInit + 1]) < 0:
        idxInit += 1
 
    x_cor1MMS = dataMMS[0][idxInit::]
    x_cor1MMS = x_cor1MMS - x_cor1MMS[0]
    y_cor1MMS = dataMMS[1][idxInit::]
    
    #volMMS
    idxInit = 1
    while np.sign(dataMMS[7][idxInit] ) == np.sign(dataMMS[7][idxInit + 1] ) \
        or np.sign(dataMMS[7][idxInit + 1]) < 0:
        idxInit += 1
 
    x_cor2MMS = dataMMS[0][idxInit::]
    x_cor2MMS = x_cor2MMS - x_cor2MMS[0]
    y_cor2MMS = dataMMS[7][idxInit::]
    
    #NUM
    x_cor1 = dataNUM.t; y_cor1 = dataNUM.y[0] #disp
    x_cor2 = dataNUM.t; y_cor2 = dataNUM.y[-1] #volt
    #cutting and phase normalization solutions from NUM
    time_lapse1 = x_cor1MMS[-1]
    time_lapse2 = x_cor2MMS[-1]
    
    #dispNUM
    idxInit = int(0.9*len(y_cor1))
    while np.sign(y_cor1[idxInit] ) == np.sign(y_cor1[idxInit + 1] ) \
        or np.sign(y_cor1[idxInit + 1]) < 0:
        idxInit += 1
    idxFin = np.where(x_cor1>= x_cor1[idxInit] + time_lapse1)[0][0]
 
    x_cor1 = x_cor1[idxInit:idxFin] 
    x_cor1 = x_cor1 -  x_cor1[0]
    y_cor1 = y_cor1[idxInit:idxFin]
    
    #voltNUM
    idxInit = int(0.9*len(y_cor2) +12)
    while np.sign(y_cor2[idxInit]) == np.sign(y_cor2[idxInit + 1] ) \
        or np.sign(y_cor2[idxInit + 1] ) < 0:
        idxInit += 1
    idxFin = np.where(x_cor2>= x_cor2[idxInit] + time_lapse2)[0][0] 
    x_cor2 = x_cor2[idxInit:idxFin] 
    x_cor2 = x_cor2 -  x_cor2[0]
    y_cor2 = y_cor2[idxInit:idxFin]
    
    
    plt.figure()
    plt.suptitle("MMS vs NUM")

    plt.subplot(2,1,1)
    plt.plot(x_cor1MMS, y_cor1MMS,'b-', label = "MMS")
    plt.plot(x_cor1  ,y_cor1,'go', markersize = 3, label = "RK")
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(x_cor2MMS, y_cor2MMS,'b-', label = "MMS")
    plt.plot(x_cor2  ,y_cor2,'go', markersize = 3, label = "RK")
    plt.xlabel('Time [s]')
    plt.ylabel('Generation [V]')
    plt.legend(loc = "upper right")
    plt.grid()
    plt.tight_layout()
    
    if save_plots is True:
        plt.savefig("output/png/" + namePlot + "MMSvsNUM", format = 'png')
        plt.savefig("output/pdf/" + namePlot + "MMSvsNUM", format = 'pdf')
        tikzplotlib.save("output/tex/" + namePlot + "MMSvsNUM.tex")
    
    
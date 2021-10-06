import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def MultipleScales(a,parsLoop):

    """
    Gives frequencies values and voltage values for a giving amplitud value. If
    a is a list the output are 4 list
    OUTPUT: freq1 and freq2 are the left and right solutions respect of the
    peak value (x-values). volt1 and volt2 are the voltage values respect to
    freq1 and freq2.
    """
    
    mu1 = parsLoop['CC']
    mu2 = parsLoop['CCC']
    theta = parsLoop['theta']
    thetaG = parsLoop['thetaG']
    thetaCp = parsLoop['thetaC']
    Kcp = parsLoop['Kc']
    alpha3 = parsLoop['Kg']
    psi = parsLoop['psi']
    psiG = parsLoop['psiG']
    psiC = parsLoop['psiC']
    g = parsLoop['FF']
    wn = parsLoop['wn']
    Cp = parsLoop['Cp']
    R = parsLoop['Rl']

    chi = parsLoop['MMM']
    alpha2 = parsLoop['Kgq']
    Kcq = parsLoop['Kcq']
    thetaCq = parsLoop['thetaCq']
    thetaGq = parsLoop['thetaGq']
    psiCq = parsLoop['psiCq']
    
    eta = -1 * np.arctan( Cp * R * wn )

    mu1eff = ( 1/2 * mu1 + ( -1/2 * R * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * theta * psi + -1/2 * R * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * thetaGq * psi ) )
    mu2eff = ( 4/3 * ( np.pi )**( -1 ) * wn * mu2 + 2/3 * ( np.pi )**( -1 ) * ( wn )**( -1 ) * thetaCp * np.sin( eta ) )
    K1eff = ( 1/2 * Cp * ( R )**( 2 ) * wn * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * theta * psi + 1/2 * Cp * ( R )**( 2 ) * wn * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * thetaGq * psi )
    K2eff = ( 4/3 * Kcp * ( np.pi )**( -1 ) * ( wn )**( -1 ) + 4/3 * ( np.pi )**( -1 ) * ( wn )**( -1 ) * thetaCp * np.cos( eta ) )
    K3eff = ( -5/12 * ( wn )**( -3 ) * ( alpha2 )**( 2 ) + ( 3/8 * ( wn )**( -1 ) * alpha3 + ( 5/24 * ( wn )**( -1 ) * alpha2 * chi + -1/4 * wn * ( chi )**( 2 ) ) ) )
    Feff = 1/4 * ( g )**( 2 ) * ( wn )**( -2 )
    
    freq1 = 1/2 * ( a )**( -2 ) * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( -1 * K1eff + ( a * ( K2eff + a * K3eff ) + wn ) ) + -1 * ( -1 * ( a )**( 2 ) * ( -1 * Feff + ( a )**( 2 ) * ( ( mu1eff + a * mu2eff ) )**( 2 ) ) )**( 1/2 ) )
    freq2 = 1/2 * ( a )**( -2 ) * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( -1 * K1eff + ( a * ( K2eff + a * K3eff ) + wn ) ) + ( -1 * ( a )**( 2 ) * ( -1 * Feff + ( a )**( 2 ) * ( ( mu1eff + a * mu2eff ) )**( 2 ) ) )**( 1/2 ) )

    K = 1/3 * ( np.pi )**( -1 ) * ( ( a )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * ( 16 * ( a )**( 2 ) * ( psiC )**( 2 ) + ( 24 * ( ( a )**( 2 ) )**( 1/2 ) * np.pi * psiC * psi + 9 * ( np.pi )**( 2 ) * ( psi )**( 2 ) ) ) )**( 1/2 )
    volt1 = abs(K)
    volt2 = abs(K)
    
    gamma = np.arcsin(( -1 * a * ( g )**( -1 ) * wn * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * mu1 + ( -1 * a * ( Cp )**( 2 ) * ( g )**( -1 ) * ( R )**( 2 ) * ( wn )**( 3 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * mu1 + ( -8/3 * ( a )**( 2 ) * ( g )**( -1 ) * ( np.pi )**( -1 ) * ( wn )**( 2 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * mu2 + ( -8/3 * ( a )**( 2 ) * ( Cp )**( 2 ) * ( g )**( -1 ) * ( np.pi )**( -1 ) * ( R )**( 2 ) * ( wn )**( 4 ) * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * mu2 + a * ( g )**( -1 ) * R * wn * ( ( 1 + ( Cp )**( 2 ) * ( R )**( 2 ) * ( wn )**( 2 ) ) )**( -1 ) * theta * psi ) ) ) ))
        
    return freq1, freq2, volt1, volt2, gamma

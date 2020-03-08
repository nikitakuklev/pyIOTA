import numpy as np
import scipy.constants


def calc_octupole_strengths(current: float, energy: float):
    """
    Calculated quasi-integrable insert strenth K3 distibution for given central current and beam energy
    :param current: 
    :param energy: 
    :return:
    """
    current = current / 1000 #Amps
    # ! phase advance over straight section
    mu0 = 0.3
    # ! length of the straight section
    l0 = 1.8
    # ! number of nonlinear elements
    nn = 17
    # ! (m^-1) strength parameter of octupole potential (for 1A/150MeV strength by default)
    alpha = 10000
    # ! dimentional parameter of nonlinear lens
    cn = 0.01
    # ! cut at multipole with power. (0) no cut, (1) quadrupole only, (3) quad+oct, (4) octupole only
    ncut = 4
    # ! type of magnet (0) thin, (1) thick, only works for octupoles (ncut=4)
    otype = 1
    # ! length of octupole for thick option. must be < l0/nn
    olen = 0.07
    # ! extra margin at the start and end of insert
    # margin = 0.0575;                      # ! extra margin at the start and end of insert
    margin = 0.022375
    # ! equivalent strength of nonlinear lens, after matching expansion terms
    tn = olen * nn / l0 * 3 / 8 * (cn ** 2) * alpha
    musect = mu0 + 0.5
    f0 = l0 / 4.0 * (1.0 + 1.0 / np.tan(np.pi * mu0) ** 2)
    betae = l0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    alfae = l0 / 2.0 / f0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    betas = l0 * (1 - l0 / 4.0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr = np.zeros(nn)
    i = np.arange(1, nn + 1)
    sn = margin + (l0 - 2 * margin) / nn * (i - 0.5)
    bn = l0 * (1 - sn * (l0 - sn) / l0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr.append(bn)
    scaling_arr = 1 / (bn ** 3)
    # March 2020 - gradient change to 0.75kG/cm^3 @1A, based on revised FEMM/harmonic measurements
    # cal_factor means K3 in m^-4 for 1A current in central octupole
    cal_factor = (0.75 / 10 * 100 * 100 * 100) / (energy / (scipy.constants.c * 1e-6))
    # bn, scaling_arr
    currents = current * cal_factor * scaling_arr / max(scaling_arr)
    return currents, bn


def calc_NL_strengths(t:float, energy: float):
    """
    Calculated NL distibution
    :param current:
    :param energy:
    :return:
    """
    # ! phase advance over straight section
    mu0 = 0.3
    # ! length of the straight section
    l0 = 1.8
    # ! number of nonlinear elements
    nn = 18
    # ! (m^-1) strength parameter of octupole potential (for 1A/150MeV strength by default)
    alpha = 10000
    # ! dimentional parameter of nonlinear lens
    cn = 0.008105461952
    # ! cut at multipole with power. (0) no cut, (1) quadrupole only, (3) quad+oct, (4) octupole only
    ncut = 4
    # ! type of magnet (0) thin, (1) thick, only works for octupoles (ncut=4)
    otype = 1
    # ! length of octupole for thick option. must be < l0/nn
    olen = 0.075
    # ! extra margin at the start and end of insert
    # margin = 0.0575;                      # ! extra margin at the start and end of insert
    margin = 0
    # ! equivalent strength of nonlinear lens, after matching expansion terms
    tn = olen * nn / l0 * 3 / 8 * (cn ** 2) * alpha
    musect = mu0 + 0.5
    f0 = l0 / 4.0 * (1.0 + 1.0 / np.tan(np.pi * mu0) ** 2)
    betae = l0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    alfae = l0 / 2.0 / f0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    betas = l0 * (1 - l0 / 4.0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr = np.zeros(nn)
    i = np.arange(1, nn + 1)
    sn = margin + (l0 - 2 * margin) / nn * (i - 0.5)
    bn = l0 * (1 - sn * (l0 - sn) / l0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
    # beta_arr.append(bn)
    knn = l0 * t / nn / bn
    cnll = cn*np.sqrt(bn)
    knll = knn*(cnll**2)
    k1 = knn*2
    #scaling_arr = 1 / (bn ** 3)
    # March 2020 - gradient change to 0.75kG/cm^3 @1A, based on revised FEMM/harmonic measurements
    # cal_factor means K3 in m^-4 for 1A current in central octupole
    #cal_factor = (0.75 / 10 * 100 * 100 * 100) / (energy / (scipy.constants.c * 1e-6))
    # bn, scaling_arr
    #currents = k1/asd/L * BR/100
    I = k1/olen
    return I, k1, bn


def NL_Current_Set(t,BR=335.26,mu0=0.3,energy=0):
    if energy != 0:
        print('BR recalc')
        BR = 1000 * (energy / (scipy.constants.c * 1e-6))
    #
    #Constants
    l0 = 1.8    #Length of Straight Section`
    #cn = 0.01   #dimentional parameter
    cn = 0.008105461952
    nn = 18     #Number of Magnets
    #L = 6.5     #Mag Length [cm]
    L = 7.5  # Mag Length [cm]
    f0 = l0/4*(1+1/np.tan(np.pi*mu0)**2) #IOTA Focus k
    beta_e = l0/np.sqrt(1-(1-l0/2/f0)**2) #Beta at Entrance
    alfa_e = l0/2/f0/np.sqrt(1-(1-l0/2/f0)**2) #Alpha Function at entrance
    beta_s = l0*(1-l0/4/f0)/np.sqrt(1-(1-l0/2/f0)**2)
    #Array stuff
    i = np.arange(1,19) #Create 1-18 array
    sn = l0/nn*(i-0.5) #Magnet distance
    bn = l0*(1-sn*(l0-sn)/l0/f0)/np.sqrt(1-(1-l0/2/f0)**2) #Beta at Magnet
    knn = l0*t*cn**2/nn/bn
    cnn = cn*np.sqrt(bn)
    k1 = 2*knn/cnn**2
    GI = [0.078015314,0.08882103,0.100544596,0.114272161,0.127323918,0.142097491,0.156935916,0.168829748,0.176310294,0.176310294,0.168829748,0.156935916,0.142097491,0.127323918,0.114272161,0.100544596,0.08882103,0.078015314]
    I = k1/GI/L*BR/100
    return I, k1, bn

#I~t & I~BR & I/t ~ BR & I/((Qy/Qx)**2-1)/2
#Input Current Return T value
def NL_Cur_t(I):
    BR=335.26
    mu0=0.3
    GI = [0.078015314,0.08882103,0.100544596,0.114272161,0.127323918,0.142097491,0.156935916,0.168829748,0.176310294,0.176310294,0.168829748,0.156935916,0.142097491,0.127323918,0.114272161,0.100544596,0.08882103,0.078015314]
    #Constants
    l0 = 1.8    #Length of Straight Section
    cn = 0.01   #dimentional parameter
    nn = 18     #Number of Magnets
    L = 6.5     #Mag Length [cm]
    f0 = l0/4*(1+1/np.tan(np.pi*mu0)**2) #IOTA Focus k
    beta_e = l0/np.sqrt(1-(1-l0/2/f0)**2) #Beta at Entrance
    alfa_e = l0/2/f0/np.sqrt(1-(1-l0/2/f0)**2) #Alpha Function at entrance
    beta_s = l0*(1-l0/4/f0)/np.sqrt(1-(1-l0/2/f0)**2)
    i = np.arange(1,19) #Create 1-18 array
    sn = l0/nn*(i-0.5) #Magnet distance
    bn = l0*(1-sn*(l0-sn)/l0/f0)/np.sqrt(1-(1-l0/2/f0)**2) #Beta at Magnet
    knn = l0*cn**2/nn/bn
    cnn = cn*np.sqrt(bn)
    k1 = 2*knn/cnn**2
    t = I/(k1/GI/L*BR/100)
    return t[9]
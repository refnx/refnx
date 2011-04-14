import numpy as np

kMassNeutron = 1.6749286e-27
kPlanck = 6.62606896e-34
#(mm/us)
kPlanck_over_MN = kPlanck/kMassNeutron

def to_q(omega, lamda):
    """
    Calculate Q from angle of incidence and wavelength
    """
    return 4 * np.pi * np.sin(omega) / lamda
    
def tof_to_lambda(TOF, distance):
    """
    Convert TOF to wavelength.
    distance = mm
    TOF = us
    return Angstom
    """
    return kPlanck_over_MN * TOF / distance * 1.e7
    
def lambda_to_tof(lamda, distance):	
	return  1.e-7 * lamda * distance / kPlanck_over_MN
    
def to_qzqy(omega, twotheta, lamda):
    """
    Calculate Qz and Qy from angle of incidence, twotheta and wavelength
    """
    return 2 * np.pi * (np.sin(twotheta - omega) + np.sin(omega))/lamda, 2 * np.pi * (np.cos(twotheta - omega) - np.cos(omega))/lamda
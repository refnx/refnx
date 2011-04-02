import numpy as np

kMassNeutron = 1.6749286e-27
kPlanck = 6.62606896e-34
kMP = 3.95603

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
    return kMP * TOF / distance
    
def lambda_to_tof(lamda, distance):	
	return  lamda * distance / kMP
    
def to_qzqy(omega, twotheta, lamda):
    """
    Calculate Qz and Qy from angle of incidence, twotheta and wavelength
    """
    return 2 * np.pi * (np.sin(twotheta - omega) + np.sin(omega))/lamda, 2 * np.pi * (np.cos(twotheta - omega) - np.cos(omega))/lamda
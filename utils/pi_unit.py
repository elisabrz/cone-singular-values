import numpy as np

def compute_angle_from_value(value):
    """
    Compute the angle in radians from the cosine value.
    
    Parameters
    ----------
    value : float
        Cosine of the angle.
        
    Returns
    -------
    angle : float
        Angle in units of pi (angle/pi).
    """
    # Ensure the value is within the valid range for arccos
    value = np.clip(value, -1, 1)
    angle = np.arccos(value) / np.pi
    return angle
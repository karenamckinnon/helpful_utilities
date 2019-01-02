import numpy as np


def RH_from_Td_T(Td, T):
    """Calculate the relative humidity value using the Magnus approximation.

    Parameters
    ----------
    Td : float or numpy array
        Dew point temperature in Celsius
    T : float or numpy array
        Temperature in Celsius

    Returns
    -------
    RH : float or numpy array
        The relative humidity, [0, 100]

    """

    return 100*np.exp(17.27*Td/(Td + 237.3))/np.exp(17.27*T/(T + 237.3))


def humidex_from_Td_T(Td, T):
    """Calculate the humidex value perCanada's Atmospheric Environment Service.

    Parameters
    ----------
    Td : float or numpy array
        Dew point temperature in Celsius
    T : float or numpy array
        Temperature in Celsius

    Returns
    -------
    humidex : float or numpy array
        Humidex index

    """

    return T + 0.5555*(6.11*np.exp(5417.7530*(1/273.16 - 1/(273.15 + Td)))-10)


def q_from_Td_p(Td, p):
    """Calculate the specific humidity from dewpoint and pressure using standard approximations.

    Parameters
    ----------
    Td : float or numpy array
        Dew point temperature in Celsius
    p : float or numpy array
        Surface pressure in mbar

    Returns
    -------
    e : float or numpy array
        The specific humidity in g/kg

    """

    ea = 6.108*np.exp(17.27*Td/(Td + 237.3))  # mbar
    q = (0.622*ea)/(p - (0.378*ea))  # kg/kg
    q *= 1e3  # g/kg

    return q


def VPD_from_Td_T(Td, T):
    """Calculate the vapor pressure deficit (VPD) value temperature and relative humidity using standard approx.

    Parameters
    ----------
    Td : float or numpy array
        Dew point temperature in Celsius
    T : float or numpy array
        Temperature in Celsius

    Returns
    -------
    VPD : float or numpy array
        Vapor pressure deficit

    """

    # http://www.fao.org/docrep/X0490E/x0490e07.htm
    es = 0.6108*np.exp(17.27*T/(T + 237.3))
    ea = 0.6108*np.exp(17.27*Td/(Td + 237.3))
    VPD = es - ea

    return VPD


def C_to_F(temp_C):
    """Convert Celsius to Fareinheit."""

    return 32 + 9/5*temp_C


def F_to_C(temp_F):
    """Convert Fareinheit to Celsius."""

    return (temp_F - 32)*5/9


def HI_from_T_RH(T, RH):
    """Calculate the heat index from temperature and relative humidity using the Rothfusz regression.

    Source: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    Original equation is in Farenheit -- have to convert from Celsius

    Parameters
    ----------
    T : float or numpy array
        Temperature in Celsius
    RH : float or numpy array
        Relative humidity in percent

    Returns
    -------
    HI : float or numpy array
        Heat index

    """

    T_F = C_to_F(T)
    if np.max(RH) < 1:  # fraction not percent
        RH *= 100

    # Calculate simple version first
    HI = 0.5*(T_F + 61.0 + ((T_F - 68.0)*1.2) + (RH*0.094))

    # Complex version used if HI > 80
    HI2 = (-42.379 + 2.04901523*T_F + 10.14333127*RH - .22475541*T_F*RH - .00683783*T_F**2 - .05481717*RH**2 +
           .00122874*T_F**2*RH + .00085282*T_F*RH**2 - .00000199*T_F**2*RH**2)

    idx_correction1 = ((RH < 13) & (T_F >= 80) & (T_F <= 112))
    correction = (13 - RH)/4*np.sqrt((17 - np.abs(T_F - 95))/17)
    HI2[idx_correction1] -= correction[idx_correction1]

    idx_correction2 = ((RH > 85) & (T_F >= 80) & (T_F <= 87))
    correction = ((RH - 85)/10)*((87 - T_F)/5)
    HI2[idx_correction2] += correction[idx_correction2]

    HI[HI > 80] = HI2[HI > 80]

    return HI

import numpy as np


def Td_from_r_p(r, p):
    """Calculate the dewpoint temperature using the Magnus approximation.

    Parameters
    ----------
    r : float or numpy array
        The water vapor mixing ratio (kg/kg)
    p : float or numpy array
        Pressure (hPa)

    Returns
    -------
    Td : float or numpy array
        Dew point temperature in Celsius
    """

    # First, convert mixing ratio to vapor pressure
    E = r*p/(0.622 + r)

    a = 6.112
    b = 17.67
    c = 243.5  # deg C

    Td = c*np.log(E/a)/(b - np.log(E/a))

    return Td


def P_from_SLP_Z(SLP, Z, T):
    """Approximate station/gridpoint pressure from sea level pressure, elevation, and temperature.

    Uses the approximation of List (1963) as referenced in Willett et al (2014)

    Parameters
    ----------
    SLP : float or numpy array
        Sea level pressure (hPa)
    Z : float or numpy array
        Elevation (meters)
    T : float or numpy array
        Temperature (Kelvin)

    Returns
    -------
    P : float or numpy array
        Approximated station pressure (hPa)
    """

    P = SLP*(T/(T + 0.0065*Z))**5.625

    return P


def Td_from_RH_T(RH, T):
    """Calculate the dewpoint temperature using the Magnus approximation.

    Parameters
    ----------
    RH : float or numpy array
        The relative humidity, [0, 100]
    T : float or numpy array
        Temperature in Celsius

    Returns
    -------
    Td : float or numpy array
        Dew point temperature in Celsius
    """

    b = 17.67
    c = 243.5  # deg C
    gamma = np.log(RH/100) + b*T/(c + T)
    Td = c*gamma/(b - gamma)

    return Td


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
        Humidex index, deg C

    """

    return T + 0.5555*(6.11*np.exp(5417.7530*(1/273.16 - 1/(273.15 + Td)))-10)


def e_from_Td_T_p(Td, T, p):
    """Calculate the vapor pressure from dewpoint, temperature and pressure.

    Temperature is used to determine whether vapor pressure needs to be calculated with respect to ice.
    Using the "quick and dirty one-thirds rule" to calculate wet bulb temperature.
    This could be improved, but right now my focus is summer so won't be very relevant.

    Parameters
    ----------
    Td : float or numpy array
        Dew point temperature in Celsius
    T : float or numpy array
        Temperature in Celsius
    p : float or numpy array
        Station pressure in mbar (or hPa)

    Returns
    -------
    e : float or numpy array
        Vapor pressure with respect to water or ice (hPa)
    """

    Tw_est = 2/3*T + 1/3*Td  # estimate of wetbulb temperature
    fw = 1 + 7e-4 + 3.46e-6*p
    fi = 1 + 3e-4 + 4.18e-6*p

    ew = 6.1121*fw*np.exp((Td*(18.729 - (Td/227.3)))/(257.87 + Td))
    ei = 6.1115*fi*np.exp((Td*(23.036 - (Td/333.7)))/(279.82 + Td))

    e = ew.copy()
    e[Tw_est < 0] = ei[Tw_est < 0]

    return e


def q_from_e_p(e, p):
    """Calculate the specific humidity from vapor pressure and pressure following Willett et al (2014).

    Parameters
    ----------
    e : float or numpy array
        Vapor pressure (mbar or hPa)
    p : float or numpy array
        Station pressure in mbar (or hPa)

    Returns
    -------
    q : float or numpy array
        The specific humidity in g/kg

    """

    q = 1000*(0.622*e/(p - 0.378*e))

    return q


def stp_from_slp(slp, T, z):
    """Calculate station pressure from sea level pressure, temperature, and height following Willett et al (2014).

    Parameters
    ----------
    slp : float or array
        Sea level pressure (hPa)
    T : float or array
        Temperature (C)
    z : float
        Station elevation

    Returns
    -------
    stp : float or array
        Estimate of pressure at the location of the station
    """

    T_K = T + 273.15  # temperature in Kelvin
    stp = slp*(T_K / (T_K + 0.0065*z))**5.625

    return stp


def q_from_GSOD_vars(df, z):
    """Calculate the specific from available variables in GSOD.

    Essentially a wrapper function for a bunch of conversions.
    Pressure data is intermittent, so using seasonal cycle as best estimate

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing GSOD data for a given station
    z : float
        Elevation (m) for the station

    Returns
    -------
    df : pandas dataframe
        Original dataframe with additional column containing q
    """

    # TODO: fix this script!! Deal with pressure via 20CR
    # Check if there are missing station pressures
    missing_stp = np.sum(np.isnan(df['stp'].values))

    if missing_stp > 0:
        # Calculate station pressure
        stp = stp_from_slp(df['slp'].values, df['temp'].values, z)

    # If we have station pressure measurements, confirm consistency


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

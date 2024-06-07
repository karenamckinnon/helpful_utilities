"""
Helper functions for downloading various types of station data.
GSOD functions are modified from https://github.com/tagomatech/ETL/blob/master/gsod/gsod.py
"""

import gzip
import requests
import re
import numpy as np
import pandas as pd
from datetime import datetime


def station_search_GSOD(options):
    '''
    Return a pandas dataframe containing the metadata for GSOD stations.

    Parameters
    ----------
    options : dict
        Dictionary containing filters for GSOD stations.
        e.g. {'ctry': "'US'", 'begin': 'datetime(1948, 1, 1)', 'end': 'datetime(2017, 12, 31)'}

    Returns
    -------
    isd_hist : pandas dataframe
        Contains metadata for selected stations

    '''

    try:
        url = 'ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv'
        df_mapping = {'USAF': str,
                      'WBAN': str,
                      'STATION NAME': str,
                      'CTRY': str,
                      'STATE': str,
                      'ICAO': str,
                      'LAT': float,
                      'LON': float,
                      'ELEV': float,
                      'BEGIN': str,
                      'END': str}
        date_parser = ['BEGIN', 'END']
        isd_hist = pd.read_csv(url,
                               dtype=df_mapping,
                               parse_dates=date_parser)

        # Rename 'STATION NAME' to 'STATION_NAME'
        isd_hist = isd_hist.rename(index=str, columns={'STATION NAME': 'STATION_NAME'})

        # Merge 'USAF' and 'WBAN'
        isd_hist['station_id'] = isd_hist.USAF + '-' + isd_hist.WBAN

        # Get rid of useless columns
        isd_hist = isd_hist.drop(['USAF', 'WBAN', 'ICAO'], axis=1)

        # Headers to lower case
        isd_hist.columns = isd_hist.columns.str.lower()

        acc = []
        for k, v in options.items():
            if k == 'begin':
                sign = '<'
            elif k == 'end':
                sign = '>'
            else:
                sign = '='

            if isinstance(v, list):
                acc.append('{} '.format(k) + sign + '= {} & '.format(v))
            else:
                acc.append('{} '.format(k) + sign + '= {} & '.format(''.join(v)))

        return isd_hist.query(re.sub('(?=.*)&.$', '', ''.join(acc)))

    except Exception as e:
        print(e)


def get_data_GSOD(station=None, start=datetime.now().year, end=datetime.now().year):
    '''
    Get weather data from the NCDC site, and return as dataframe.

    Parameters
    ----------
    station : str
        Numeric identifier for station. Use form returned by station_search function.
    start : int
        Year to start getting data (inclusive)
    end : int
        Year to finish getting data (inclusive)

    Returns
    -------
    big_df : pandas dataframe
        Dataframe containing the weather information for the specified station and years.

    '''
    big_df = pd.DataFrame()

    for year in range(start, end+1):

        # Define URL
        url = 'http://www1.ncdc.noaa.gov/pub/data/gsod/' + str(year) + '/' + str(station) \
            + '-' + str(year) + '.op.gz'

        # Define data stream
        stream = requests.get(url)

        # Unzip on-the-fly
        try:
            decomp_bytes = gzip.decompress(stream.content)
        except OSError:  # case where file does not exist
            continue

        data = decomp_bytes.decode('utf-8').split('\n')

        '''
        Data manipulations and ordering
        '''
        # Remove start and end
        data.pop(0)  # Remove first line header
        data.pop()  # Remove last element

        # Define lists
        (stn, wban, date, temp, temp_c, dewp, dewp_c,
         slp, slp_c, stp, stp_c, visib, visib_c,
         wdsp, wdsp_c, mxspd, gust, max_temp, max_temp_flag, min_temp, min_temp_flag,
         prcp, prcp_f, sndp, f, r, s, h, th, tr) = ([] for i in range(30))

        # Fill in lists
        for i in range(0, len(data)):
            stn.append(data[i][0:6])
            wban.append(data[i][7:12])
            date.append(data[i][14:22])
            temp.append(data[i][25:30])
            temp_c.append(data[i][31:33])
            dewp.append(data[i][36:41])
            dewp_c.append(data[i][42:44])
            slp.append(data[i][46:52])      # Mean sea level pressure
            slp_c.append(data[i][53:55])
            stp.append(data[i][57:63])      # Mean station pressure
            stp_c.append(data[i][64:66])
            visib.append(data[i][68:73])
            visib_c.append(data[i][74:76])
            wdsp.append(data[i][78:83])
            wdsp_c.append(data[i][84:86])
            mxspd.append(data[i][88:93])
            gust.append(data[i][95:100])
            max_temp.append(data[i][103:108])
            max_temp_flag.append(data[i][108])
            min_temp.append(data[i][111:116])
            min_temp_flag.append(data[i][116])
            prcp.append(data[i][118:123])
            prcp_f.append(data[i][123])
            sndp.append(data[i][125:130])   # Snow depth in inches to tenth
            f.append(data[i][132])          # Fog
            r.append(data[i][133])          # Rain or drizzle
            s.append(data[i][134])          # Snow or ice pallet
            h.append(data[i][135])          # Hail
            th.append(data[i][136])         # Thunder
            tr.append(data[i][137])         # Tornado or funnel cloud

        '''
        Replacements
        min_temp_flag & max_temp_flag
        blank   : explicit => e
        *       : derived => d
        '''
        max_temp_flag = [re.sub(pattern=' ', repl='e', string=x) for x in max_temp_flag]  # List comprenhension
        max_temp_flag = [re.sub(pattern=r"\*", repl='d', string=x) for x in max_temp_flag]

        min_temp_flag = [re.sub(pattern=' ', repl='e', string=x) for x in min_temp_flag]
        min_temp_flag = [re.sub(pattern=r"\*", repl='d', string=x) for x in min_temp_flag]

        '''
        Create dataframe & cleanse data
        '''
        # Create intermediate matrix
        mat = np.matrix(data=[stn, wban, date, temp, temp_c, dewp, dewp_c,
                              slp, slp_c, stp, stp_c, visib, visib_c,
                              wdsp, wdsp_c, mxspd, gust, max_temp, max_temp_flag, min_temp, min_temp_flag,
                              prcp, prcp_f, sndp, f, r, s, h, th, tr]).T

        # Define header names
        headers = ['stn', 'wban', 'date', 'temp', 'temp_c', 'dewp', 'dewp_c',
                   'slp', 'slp_c', 'stp', 'stp_c', 'visib', 'visib_c',
                   'wdsp', 'wdsp_c', 'mxspd', 'gust', 'max_temp', 'max_temp_flag', 'min_temp', 'min_temp_flag',
                   'prcp', 'prcp_f', 'sndp', 'f', 'r', 's', 'h', 'th', 'tr']

        # Set precision
        pd.set_option('precision', 3)

        # Create dataframe from matrix object
        df = pd.DataFrame(data=mat, columns=headers)

        # Replace missing values with NAs
        df = df.where(df != ' ', 9999.9)

        # Create station ids
        df['station_id'] = df['stn'].map(str) + '-' + df['wban'].map(str)
        df = df.drop(['stn', 'wban'], axis=1)

        # Convert to numeric
        df[['temp', 'temp_c', 'dewp', 'dewp_c',
            'slp', 'slp_c', 'stp', 'stp_c',
            'visib', 'visib_c', 'wdsp', 'wdsp_c',
            'mxspd',  'gust', 'max_temp',
            'min_temp', 'prcp', 'sndp']] = df[['temp', 'temp_c', 'dewp',
                                               'dewp_c', 'slp', 'slp_c', 'stp',
                                               'stp_c', 'visib', 'visib_c', 'wdsp',
                                               'wdsp_c', 'mxspd', 'gust', 'max_temp',
                                               'min_temp', 'prcp', 'sndp']].apply(pd.to_numeric)

        # Replace missing weather data with NaNs
        df = df.replace(to_replace=[99.99, 99.9, 999.9, 9999.9], value=np.nan)

        # Convert to date format
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        big_df = pd.concat([big_df, df])

    # Reset index
    big_df = big_df.reset_index()

    return big_df

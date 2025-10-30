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
from subprocess import check_call
import os
import xarray as xr


def process_ghcnd(yr_start, yr_end, ghcnd_dir='/home/data/GHCND', var_names=['TMAX', 'TMIN'], country_list=None):
    """This function will subset GHNCD dly files to ones that have sufficient coverage and, if desired, are in a
    specific set of countries.

    To update GHCND data (first), run summer_extremes/scripts/update_ghcnd_data.sh

    Parameters
    ----------
    yr_start : int
        Latest year at which a station should have data
    yr_end : int
        Earliest year in which a station can no longer have data
    ghcnd_dir : str
        Directory containing dly files
    var_names : list
        List of variable names to keep. Standard 5 vars in GHCND: PRCP, SNOW, SNWD, TMAX, TMIN
    country_list : list or None
        List of countries to save data from (FIPS country code), or None if all countries desired

    """

    f_inventory = '%s/ghcnd-inventory.txt' % ghcnd_dir
    outdir = '%s/%04i-%04i' % (ghcnd_dir, yr_start, yr_end)
    cmd = 'mkdir -p %s' % outdir
    check_call(cmd.split())

    # Pull information from inventory
    namestr = [0, 11]
    latstr = [12, 20]
    lonstr = [21, 30]
    varstr = [31, 35]
    startstr = [36, 40]
    endstr = [41, 45]
    for ct_v, this_var in enumerate(var_names):

        with open(f_inventory, 'r') as f:
            name = []
            lon = []
            lat = []
            start = []
            end = []

            for line in f:
                var = line[varstr[0]:varstr[1]]
                if (var == this_var):
                    name.append(line[namestr[0]:namestr[1]])  # station name
                    lat.append(line[latstr[0]:latstr[1]])  # station latitude
                    lon.append(line[lonstr[0]:lonstr[1]])  # station longitude
                    start.append(line[startstr[0]:startstr[1]])  # start year of station data
                    end.append(line[endstr[0]:endstr[1]])  # end year of station data

            this_dict = [{'name': name, 'lat': lat, 'lon': lon, 'start': start, 'end': end}
                         for name, lat, lon, start, end in zip(name, lat, lon, start, end)]

            if ct_v == 0:
                inventory_dict = {this_var: this_dict}
            else:
                inventory_dict[this_var] = this_dict

    for ct_v, this_var in enumerate(var_names):
        station_list = []
        lons = []
        lats = []

        for key in inventory_dict[this_var]:
            this_name = key['name']
            this_start = float(key['start'])
            this_end = float(key['end'])

            in_region = True
            if country_list is not None:
                # if subsetting to countries, set to False, then change to true if match
                in_region = False
                for c in country_list:
                    in_region = this_name[:2] == c
                    if in_region:
                        break

            if (in_region & (this_start <= yr_start) & (this_end >= yr_end)):

                # Add info for each station if not already added to the list
                if this_name not in station_list:

                    station_list.append(this_name)
                    lons.append(float(key['lon']))
                    lats.append(float(key['lat']))

    # Get data for each station
    # ------------------------------
    # Variable   Columns   Type
    # ------------------------------
    # ID            1-11   Character
    # YEAR         12-15   Integer
    # MONTH        16-17   Integer
    # ELEMENT      18-21   Character
    # VALUE1       22-26   Integer
    # MFLAG1       27-27   Character
    # QFLAG1       28-28   Character
    # SFLAG1       29-29   Character
    # VALUE2       30-34   Integer
    # MFLAG2       35-35   Character
    # QFLAG2       36-36   Character
    # SFLAG2       37-37   Character
    #   .           .          .
    #   .           .          .
    #   .           .          .
    # VALUE31    262-266   Integer
    # MFLAG31    267-267   Character
    # QFLAG31    268-268   Character
    # SFLAG31    269-269   Character
    # ------------------------------

    # These variables have the following definitions:

    # ID         is the station identification code.  Please see "ghcnd-stations.txt"
    #            for a complete list of stations and their metadata.
    # YEAR       is the year of the record.

    # MONTH      is the month of the record.

    # ELEMENT    is the element type.   There are five core elements as well as a number
    #            of addition elements.

    #            The five core elements are:

    #            PRCP = Precipitation (tenths of mm)
    #            SNOW = Snowfall (mm)
    #            SNWD = Snow depth (mm)
    #            TMAX = Maximum temperature (tenths of degrees C)
    #            TMIN = Minimum temperature (tenths of degrees C)

    date_str = pd.date_range(start='1850-01-01', end='%04i-12-31' % yr_end, freq='D')

    yearstr = [11, 15]
    monstr = [15, 17]
    varstr = [17, 21]
    datastr = [21, 269]

    for counter, this_station in enumerate(station_list):
        print(this_station)
        print('%i/%i' % (counter, len(station_list)))
        this_file = '%s/ghcnd_all/%s.dly' % (ghcnd_dir, this_station)

        if os.path.isfile(this_file):

            for this_var in var_names:
                savename = '%s/%s_%s.nc' % (outdir, this_station, this_var)
                data_vec = np.nan*np.ones(len(date_str))
                if os.path.isfile(savename):
                    continue
                with open(this_file, 'r') as f:
                    for line in f:
                        if this_var == line[varstr[0]: varstr[1]]:
                            this_year = line[yearstr[0]: yearstr[1]]

                            if float(this_year) >= 1850:  # only keeping data back to 1850
                                mon = line[monstr[0]: monstr[1]]  # the month of data

                                data = line[datastr[0]: datastr[1]]  # the data

                                days = [data[i*8:i*8+8] for i in np.arange(0, 31, 1)]
                                mflag = [days[i][5] for i in np.arange(31)]  # getting the mflag
                                qflag = [days[i][6] for i in np.arange(31)]  # getting the qflag
                                values = [days[i][:5] for i in np.arange(31)]  # getting the data values
                                values_np = np.array(values).astype(int)  # converting to a numpy array

                                # set missing to NaN
                                is_missing = (values_np == -9999)
                                values_np = values_np.astype(float)
                                values_np[is_missing] = np.nan

                                # removing any that fail the quality control flag or have
                                # L = temperature appears to be lagged with respect to reported hour of observation
                                is_bad = (np.array(qflag) != ' ') | (np.array(mflag) == 'L')
                                values_np[is_bad] = np.nan

                                date_idx = (date_str.month == int(mon)) & (date_str.year == int(this_year))
                                data_vec[date_idx] = values_np[:np.sum(date_idx)]/10  # change to degrees Celsius

                # If no data
                if np.isnan(data_vec).all():
                    continue
                # Remove starting and ending NaNs
                start_idx = np.where(~np.isnan(data_vec))[0][0]
                end_idx = np.where(~np.isnan(data_vec))[0][-1] + 1

                new_date_str = date_str[start_idx:end_idx]
                data_vec = data_vec[start_idx:end_idx]

                # Save data
                this_da = xr.DataArray(data_vec, dims='time', coords={'time': new_date_str})
                this_da['lat'] = lats[counter]
                this_da['lon'] = lons[counter]
                this_da.to_netcdf(savename)


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

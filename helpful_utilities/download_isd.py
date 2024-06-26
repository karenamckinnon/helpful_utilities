import os
from subprocess import check_call
import pandas as pd
from datetime import datetime
import numpy as np
import argparse
pd.set_option('max_columns', None)


def remove_bad_rows(df):
    """Based on ISD documentation, remove data with the following sources, report types, call signs, and QC.

    ISD docs: https://www1.ncdc.noaa.gov/pub/data/noaa/isd-format-document.pdf

    # remove SOURCE:
    # 2: failed cross checks
    # A: failed cross checks
    # B: failed cross checks
    # O: Summary observation created by NCEI using hourly observations that
    #    may not share the same data source flag
    # 9: missing

    # remove REPORT_TYPE
    # 99999: missing

    # remove CALL_SIGN
    # 99999: missing

    # remove QUALITY_CONTROL
    # V01: no quality control
    """
    bad_idx = ((df['SOURCE'].astype(str) == '2') |
               (df['SOURCE'].astype(str) == 'A') |
               (df['SOURCE'].astype(str) == 'B') |
               (df['SOURCE'].astype(str) == 'O') |
               (df['SOURCE'].astype(str) == '9') |
               (df['REPORT_TYPE'].astype(str) == '99999') |
               (df['QUALITY_CONTROL'].astype(str) == 'V010'))

    return df[~bad_idx]


def remove_bad_vals(df, varname):
    """
    Remove values with questionable flags, based on ISD documentation.

    ISD docs: https://www1.ncdc.noaa.gov/pub/data/noaa/isd-format-document.pdf
    """
    flag = np.array([d.split(',')[-1] for d in df[varname]])
    flag = flag.astype(str)
    vals_tmp = np.array([int(d.split(',')[0]) for d in df[varname]])

    if ((varname == 'DEW') | (varname == 'TMP')):
        bad_idx = ((flag == '2') |
                   (flag == '3') |
                   (flag == '6') |
                   (flag == '7') |
                   (flag == 'A') |
                   (flag == 'C') |
                   (vals_tmp == 9999))
    elif varname == 'SLP':
        bad_idx = ((flag == '2') |
                   (flag == '3') |
                   (flag == '6') |
                   (flag == '7') |
                   (vals_tmp == 99999))

    vals = vals_tmp.astype(float)/10

    vals[bad_idx] = np.nan
    df = df.assign(**{varname: vals})

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('tmpdir', type=str, help='full path to tmp directory to store intermediate stuff')
    parser.add_argument('savedir', type=str, help='full path to directory where to save files')
    parser.add_argument('start_date', type=str, help='YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='YYYY-MM-DD')
    parser.add_argument('ctry', type=str, help='2 letter abbreviation, or list e.g. CA,US (no space between), or None')
    parser.add_argument('state', type=str, help='2 letter abbreviation, or list, or None')
    parser.add_argument('lat1', type=float, help='lower lat')
    parser.add_argument('lat2', type=float, help='upper lat')
    parser.add_argument('lon1', type=float, help='lower lon')
    parser.add_argument('lon2', type=float, help='upper lon')

    args = parser.parse_args()

    tmpdir = args.tmpdir
    savedir = args.savedir
    start_date = args.start_date
    end_date = args.end_date
    ctry = (args.ctry).split(',')
    state = (args.state).split(',')
    lat1 = args.lat1
    lat2 = args.lat2
    lon1 = args.lon1
    lon2 = args.lon2

    tmpdir = tmpdir.rstrip('/')
    savedir = savedir.rstrip('/')

    cmd = 'mkdir -p %s' % tmpdir
    check_call(cmd.split())
    cmd = 'mkdir -p %s/csv' % savedir
    check_call(cmd.split())

    # Must start at or before start year
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Get latest metadata
    url_base = 'https://www.ncei.noaa.gov/data/global-hourly/access'
    url_history = 'ftp://ftp.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
    hist_name = url_history.split('/')[-1]

    # if not os.path.isfile('%s/%s' % (savedir, hist_name)):
    cmd = 'wget -q -O %s/%s %s' % (savedir, hist_name, url_history)
    check_call(cmd.split())

    df_meta = pd.read_csv('%s/%s' % (savedir, hist_name))
    dt_begin = np.array([datetime.strptime(str(d), '%Y%m%d') for d in df_meta['BEGIN']])
    dt_end = np.array([datetime.strptime(str(d), '%Y%m%d') for d in df_meta['END']])
    idx_use_time = ((dt_begin <= start_date) & (dt_end >= end_date))
    idx_use_space = idx_use_time.copy()

    if 'None' not in ctry:
        idx_use_space = idx_use_space & (np.isin(df_meta['CTRY'], ctry))
    if 'None' not in state:
        idx_use_space = idx_use_space & (np.isin(df_meta['STATE'], state))
    idx_use_lats = (df_meta['LAT'] >= lat1) & (df_meta['LAT'] <= lat2).values
    idx_use_lons = (df_meta['LON'] >= lon1) & (df_meta['LON'] <= lon2).values
    idx_use_space = idx_use_space & idx_use_lats & idx_use_lons

    df_meta = df_meta.assign(BEGIN=dt_begin, END=dt_end)
    df_meta = df_meta[idx_use_space].reset_index()

    usecols = ['SOURCE', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL', 'DATE', 'ELEVATION', 'DEW', 'TMP', 'SLP']
    keepcols = ['DATE', 'DEW', 'TMP', 'SLP']
    has_data = np.ones(len(df_meta)).astype(bool)

    for ct, row in df_meta.iterrows():
        savename = '%s/csv/%06d-%05d.csv' % (savedir, int(row['USAF']), int(row['WBAN']))
        if os.path.isfile(savename):
            continue

        print('%i/%i' % (ct, len(df_meta)))
        # download files for each year
        station_id = '%06d%05d' % (int(row['USAF']), int(row['WBAN']))
        yy1 = row['BEGIN'].year
        yy2 = row['END'].year

        # download all files
        all_df = []
        for yy in range(yy1, yy2 + 1):

            url = '%s/%i/%s.csv' % (url_base, yy, station_id)
            fname = '%s_%i.csv' % (station_id, yy)
            cmd = 'wget -q -O %s/%s %s' % (tmpdir, fname, url)
            try:
                check_call(cmd.split())
            except Exception as e:  # file not available
                print(str(e))
                continue

            df = pd.read_csv('%s/%s' % (tmpdir, fname), usecols=usecols, low_memory=False)
            elev = df.loc[0, 'ELEVATION']  # meters

            df = remove_bad_rows(df)
            df = df[keepcols]

            # convert data to float
            for varname in ('TMP', 'DEW', 'SLP'):
                df = remove_bad_vals(df, varname)

            cmd = 'rm -f %s/%s' % (tmpdir, fname)
            check_call(cmd.split())
            all_df.append(df)

        if len(all_df) > 0:
            all_df = pd.concat(all_df).reset_index()
        else:  # no files available
            has_data[ct] = False
            continue

        # Estimate station pressure (as per Willett et al 2014)
        stp = all_df['SLP']*((all_df['TMP'] + 273.15)/(all_df['TMP'] + 273.15 + 0.0065*elev))**5.625

        # Calculate q (g/kg)
        e = 6.112*np.exp((17.67*all_df['DEW'])/(all_df['DEW'] + 243.5))
        q = 1000*(0.622 * e)/(stp - (0.378 * e))  # g/kg

        all_df = all_df.assign(STP=stp, Q=q)
        dt = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in all_df['DATE']]
        all_df.index = dt

        resampler = all_df.resample('D')
        count = resampler.count()
        avg_df = resampler.mean()
        max_df = resampler.max()
        min_df = resampler.min()
        avg_df[count < 4] = np.nan
        max_df[count < 4] = np.nan
        min_df[count < 4] = np.nan

        avg_df = avg_df.assign(TMAX=max_df['TMP'])
        avg_df = avg_df.assign(TMIN=min_df['TMP'])
        avg_df = avg_df.assign(TMP_COUNT=count['TMP'])

        # get rid of extra index
        avg_df = avg_df.iloc[:, 1:]

        # Fix title of date column
        avg_df = avg_df.reset_index()
        avg_df = avg_df.rename(columns={'index': 'date'})

        # Make sure there are a reasonable number of values
        if np.sum(~np.isnan(avg_df['TMAX'])) < 100:
            has_data[ct] = False
        else:
            # Save
            avg_df.to_csv(savename, index=False)
            all_df.to_csv(savename.replace('.csv', '_hourly.csv'))

    # remove stations with no data from metadata
    df_meta = df_meta[has_data]

    # Change to lowercase headings, add station_id, and save
    df_meta.columns = [c.lower() for c in df_meta.columns]
    df_meta = df_meta.assign(station_id='%s-%s' % (df_meta['usaf'], df_meta['wban']))
    df_meta.to_csv('%s/csv/new_metadata.csv' % savedir)

#!/bin/bash

update_data=true
make_ncs=true

if $update_data
then
    ghcnd_dir=/home/data/GHCND
    cd $ghcnd_dir

    # get docs
    wget -N https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
    wget -N https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt
    wget -N https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
    wget -N https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-version.txt

    # get data
    wget https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz

    # unzip data files
    tar -xzvf ghcnd_all.tar.gz
    rm -f ghcnd_all.tar.gz
else
    echo "Not updating GHCND .dly files"
fi

# Important: the default variable is TMAX. See the function for documentation of options
# process_ghcnd(yr_start, yr_end, ghcnd_dir='/home/data/GHCND', var_names=['TMAX'], country_list=None)
if $make_ncs
then
    python -c'from helpful_utilities import stations; stations.process_ghcnd(1979, 2023)'
else
    echo "Not making netcdf files"
fi

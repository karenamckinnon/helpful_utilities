era5_scripts_dir=/home/kmckinnon/helpful_utilities/helpful_utilities/proc_ERA5
varname=surface_latent_heat_flux
calc_max=false
calc_min=false
delete_hourly=false
time_res=day  # for regridding specifically

download_data=true
aggregate_data=true
regrid_data=true

for yy in {1959..2023}
do
    echo $yy
    if $download_data
    then
        python ${era5_scripts_dir}/download_hourly_ERA5_sfc.py --year $yy --era5_var_name ${varname}
    else
        echo "Not downloading ${varname}"
    fi

    if $aggregate_data
    then
        python3.9 ${era5_scripts_dir}/aggregate_hourly.py --year $yy --era5_var_name ${varname} --calc_max ${calc_max} --calc_min ${calc_min} --delete_hourly ${delete_hourly}
    else
        echo "Not aggregating daily"
    fi

done

if $regrid_data
then
    python ${era5_scripts_dir}/regrid_1x1.py --era5_var_name ${varname} --time_resolution ${time_res}
else
    echo "Not regridding"
fi

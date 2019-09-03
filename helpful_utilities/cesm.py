import numpy as np
import xarray as xr


def get_CESM1_lsmask():
    """Quick helper function to get a CESM land mask when on Cheyenne."""

    land_file = ('/gpfs/fs1/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/lnd/proc/tseries/'
                 'monthly/NEP/b.e11.B1850C5CN.f09_g16.005.clm2.h0.NEP.040001-049912.nc')
    lsmask = xr.open_dataset(land_file)
    lsmask = ~np.isnan(lsmask.NEP[0, ...])
    lsmask = lsmask.rename('is_land')
    lsmask = lsmask.drop('time')

    return lsmask

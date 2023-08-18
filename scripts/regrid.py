import os
from glob import glob

import xarray as xr
import numpy as np
import xesmf as xe


"""
Regrid horizontally.
:param ds_in: Input xarray dataset
:param ddeg_out: Output resolution
:param method: Regridding method
:param reuse_weights: Reuse weights for regridding
:return: ds_out: Regridded dataset
"""

def regrid(ds_in, ddeg_out, method='bilinear', reuse_weights=True, cmip=False, rename=None):

    # Rename to ESMF compatible coordinates
    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})
    if cmip:
        ds_in = ds_in.drop(('lat_bnds', 'lon_bnds'))
        if hasattr(ds_in, 'plev_bnds'):
            ds_in = ds_in.drop(('plev_bnds'))
        if hasattr(ds_in, 'time_bnds'):
            ds_in = ds_in.drop(('time_bnds'))
    if rename is not None:
        ds_in = ds_in.rename({rename[0]: rename[1]})

    # Create output grid
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+ddeg_out/2, 90, ddeg_out)),
            'lon': (['lon'], np.arange(0, 360, ddeg_out)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True, reuse_weights=reuse_weights
    )

    # Hack to speed up regridding of large files
    ds_out = regridder(ds_in, keep_attrs=True).astype('float32')

    if rename is not None:
        if rename[0] == 'zg':
            ds_out['z'] *= 9.807
        if rename[0] == 'rsdt':
            ds_out['tisr'] *= 60*60
            ds_out = ds_out.isel(time=slice(1, None, 12))
            ds_out = ds_out.assign_coords({'time': ds_out.time + np.timedelta64(90, 'm')})

    # # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def regrid_a_file(file_path, save_path, ddeg_out=5.625):
    x = xr.open_dataset(file_path)
    # if 'input' in p:
    #     for v in constant_vars:
    #         x[v] = x[v].expand_dims(dim={'latitude': 96, 'longitude': 144}, axis=(1,2))
    x_regridded = regrid(x, ddeg_out, reuse_weights=False)
    x_regridded.to_netcdf(save_path)


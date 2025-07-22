### Imports, some helper functions
import numpy as np
import xarray as xr
import glob
import os
import hashlib
import dask
import gc
from tqdm import tqdm
import atexit
import pandas as pd

# stuff for running headless
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib import colormaps
from matplotlib import cm
from dask.distributed import Client, LocalCluster
import dask.config
from multiprocessing import get_context
import seaborn as sns
import pprint
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pprint import pp
from datetime import date

# let's test for now...
from xarray_cacher import *

### Data loading
# Corrected lat-lon grid will be handled separately
model_grid = xr.open_mfdataset("/data/ycheng46/NNA/data/alaska_climate_region.nc")
static_lat2d = model_grid.lat.values
static_lon2d = model_grid.lon.values % 360

# DAILY snow
snow_paths = [
    "/data/shared_data/NNA/NNA.4km.hERA5.1989.003/snow_d/*.nc",
    "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/snow_d/*.nc",
    "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/snow_d/*.nc",
]

better_h2osno_paths = [
    "/data/shared_data/NNA/NNA.4km.hERA5.1989.003/swe_d/*.nc",
    "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/swe_d/*.nc",
    "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/swe_d/*.nc",
]

snow_P, snow_H, snow_M = unpacker(snow_paths, "SNOW")
better_h2osno_P, better_h2osno_H, better_h2osno_M = unpacker(better_h2osno_paths, "H2OSNO")

### Data editing

## Constants
seconds_per_year = 365.25 * 86400
seconds_per_season = 90.25 * 86400

## Snow
total_snow_P = (snow_P) * seconds_per_year
total_snow_H = (snow_H) * seconds_per_year
total_snow_M = (snow_M) * seconds_per_year

## Better snowmelt (snow water equivalent)
# In this case, we're just grouping, we have high enough precision where we want individual days
total_better_h2osno_P = better_h2osno_P
total_better_h2osno_H = better_h2osno_H
total_better_h2osno_M = better_h2osno_M

def snow_by_gridcells(snow_ds, swe_ds, year_range=(1989, 1990), scenario_flag="P"):
    def one_year_range(year_start, year_end, scenario_type):
        print(f"starting {year_start}-{year_end}_{scenario_type}")
        # use this to deal with things like leapyears and whatnot
        def days_in_year(year):
            return date(year, 12, 31).timetuple().tm_yday
        
        max_swe = swe_ds.sel(time=slice(f"{year_end}-01-01", f"{year_end}-08-31")).max("time", skipna=True)
        min_swe = swe_ds.sel(time=slice(f"{year_start}-04-01", f"{year_start}-12-31")).min("time", skipna=True)
        max_doy = swe_ds.sel(time=slice(f"{year_end}-01-01", f"{year_end}-08-31")).idxmax("time").dt.dayofyear + days_in_year(year_start)
        min_doy = swe_ds.sel(time=slice(f"{year_start}-04-01", f"{year_start}-12-31")).idxmin("time").dt.dayofyear
        print(f"slicing complete for {year_start}-{year_end}_{scenario_type}")

        total_days = days_in_year(year_start) + days_in_year(year_end)
        doy_axis = xr.DataArray(np.arange(1, total_days+1), dims="time")
        snow_slice = snow_ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        doy_broadcast = xr.broadcast(doy_axis, snow_slice.isel(time=0))[0]

        mask = (doy_broadcast >= min_doy) & (doy_broadcast <= max_doy)
        masked_snow = snow_slice.where(mask) * 86400
        total_snow = masked_snow.sum("time", skipna=True)

        snow_ratio = (max_swe - min_swe) / total_snow
        print(f"math complete for {year_start}-{year_end}_{scenario_type}")

        output_dir = "outputs_nuevo" # change if needed

        os.makedirs(output_dir, exist_ok=True)
        out_base = f"{output_dir}/snowratio_{year_start}_{year_end}_{scenario_type}"
        # this following line is a dumb trick to give us times to sort over.
        snow_ratio = snow_ratio.expand_dims(snow_year=[f"{year_start}-{year_end}"])
        snow_ratio.name = "snow_ratio"
        snow_ratio.to_netcdf(f"{out_base}.nc")
        print(f"netcdf export complete for {year_start}-{year_end}_{scenario_type}")

        plt.figure(dpi=300)
        snow_ratio.plot(vmax=1)
        plt.savefig(f"{out_base}.png")
        plt.close()
        print(f"finished {year_start}-{year_end}_{scenario_type}")

        return snow_ratio

    if isinstance(year_range, tuple):
        years = [(y, y + 1) for y in range(year_range[0], year_range[1])]
    elif isinstance(year_range, int):
        years = [(year_range, year_range + 1)]
    else:
        raise TypeError("year_range must be int or tuple")

    lazy_results = [dask.delayed(one_year_range)(y0, y1, scenario_flag) for (y0, y1) in years]
    computed = dask.compute(*lazy_results)
    return xr.concat(computed, dim="snow_year").assign_coords(snow_year=[f"{y0}-{y1}" for (y0, y1) in years])

def main():
    cluster = LocalCluster(
        n_workers=80,
        threads_per_worker=4,
        memory_limit=0,
        processes=True,
        nanny=False
    )
    client = Client(cluster)
    print("dashboard:", client.dashboard_link)

    def shutdown():
        print("shutting down cluster...")
        client.close()
        cluster.close()
    atexit.register(shutdown)

    # you have to uncomment one at a time due to stupid ulimit
    # make sure to raise it if you want to run this code
    
    final_stack_P = snow_by_gridcells(snow_P, total_better_h2osno_P, (1990, 2020), "P")
    final_stack_H = snow_by_gridcells(snow_H, total_better_h2osno_H, (2034, 2064), "H")
    final_stack_M = snow_by_gridcells(snow_M, total_better_h2osno_M, (2034, 2064), "M")
    #final_stack.to_netcdf("test_snow_all.nc") # as far as I can tell, this crushes the ulimit

    shutdown()
    print("done writing netcdf file.")
    gc.collect()

if __name__ == "__main__":
    main()

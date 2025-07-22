### Imports, some helper functions
import numpy as np
import xarray as xr
import glob
import os
import hashlib
import dask
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from dask.distributed import Client, LocalCluster
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
snow_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/snow_m/*.nc",
              "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/snow_m/*.nc",
              "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/snow_m/*.nc" ]

rain_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/rain_m/*.nc", 
              "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/rain_m/*.nc", 
              "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/rain_m/*.nc" ]

temp_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/tsa_m/*.nc", 
              "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/tsa_m/*.nc", 
              "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/tsa_m/*.nc" ]

better_h2osno_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/swe_d/*.nc",
                        "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/swe_d/*.nc",
                        "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/swe_d/*.nc" ]

qrunoff_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/qrunoff_m/*.nc",
                 "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/qrunoff_m/*.nc",
                 "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/qrunoff_m/*.nc" ]

snow_P, snow_H, snow_M = unpacker(snow_paths, "SNOW")
rain_P, rain_H, rain_M = unpacker(rain_paths, "RAIN")
temp_P, temp_H, temp_M = unpacker(temp_paths, "TSA")
better_h2osno_P, better_h2osno_H, better_h2osno_M = unpacker(better_h2osno_paths, "H2OSNO")
qrunoff_P, qrunoff_H, qrunoff_M = unpacker(qrunoff_paths, "QRUNOFF")

### Data editing
## Constants
seconds_per_year = 365.25 * 86400
seconds_per_season = 90.25 * 86400

## Helpers for our cache functions
def compute_annual_generic(ds):
    return ds.groupby("time.year").mean(dim="time", skipna=True, keep_attrs=True)

def compute_annual_runoff(ds):
    return ds.groupby("time.year").mean(dim="time", skipna=True, keep_attrs=True)

def compute_geo_average_future(ds):
    return ds.mean(dim=("lat", "lon"), skipna=True, keep_attrs=True).sel(year=slice(2034, 2064))

def compute_geo_average_historic(ds):
    return ds.mean(dim=("lat", "lon"), skipna=True, keep_attrs=True).sel(year=slice(1990, 2020))

def compute_layer_average_future(ds):
    return ds.mean(dim=("lat", "lon"), skipna=True, keep_attrs=True).sum(dim="levsoi").sel(year=slice(2034, 2064))

def compute_layer_average_historic(ds):
    return ds.mean(dim=("lat", "lon"), skipna=True, keep_attrs=True).sum(dim="levsoi").sel(year=slice(1990, 2020))

## Precipitation
total_precip_P = (rain_P + snow_P) * seconds_per_year
total_precip_H = (rain_H + snow_H) * seconds_per_year
total_precip_M = (rain_M + snow_M) * seconds_per_year

total_precip_P.name = "PRECIP_P"
total_precip_H.name = "PRECIP_H"
total_precip_M.name = "PRECIP_M"

annual_precip_P = cache_xarray(compute_annual_generic, total_precip_P, name_hint="annual_precip_P")
annual_precip_H = cache_xarray(compute_annual_generic, total_precip_H, name_hint="annual_precip_H")
annual_precip_M = cache_xarray(compute_annual_generic, total_precip_M, name_hint="annual_precip_M")

## Snow
total_snow_P = (snow_P) * seconds_per_year
total_snow_H = (snow_H) * seconds_per_year
total_snow_M = (snow_M) * seconds_per_year

annual_snow_P = cache_xarray(compute_annual_generic, total_snow_P, name_hint="annual_snow_P")
annual_snow_H = cache_xarray(compute_annual_generic, total_snow_H, name_hint="annual_snow_H")
annual_snow_M = cache_xarray(compute_annual_generic, total_snow_M, name_hint="annual_snow_M")

## Rain
total_rain_P = (rain_P) * seconds_per_year
total_rain_H = (rain_H) * seconds_per_year
total_rain_M = (rain_M) * seconds_per_year

annual_rain_P = cache_xarray(compute_annual_generic, total_rain_P, name_hint="annual_rain_P")
annual_rain_H = cache_xarray(compute_annual_generic, total_rain_H, name_hint="annual_rain_H")
annual_rain_M = cache_xarray(compute_annual_generic, total_rain_M, name_hint="annual_rain_M")

## Runoff
total_qrunoff_P = qrunoff_P * seconds_per_year
total_qrunoff_H = qrunoff_H * seconds_per_year
total_qrunoff_M = qrunoff_M * seconds_per_year

annual_qrunoff_P = cache_xarray(compute_annual_runoff, total_qrunoff_P, name_hint="annual_qrunoff_P")
annual_qrunoff_H = cache_xarray(compute_annual_runoff, total_qrunoff_H, name_hint="annual_qrunoff_H")
annual_qrunoff_M = cache_xarray(compute_annual_runoff, total_qrunoff_M, name_hint="annual_qrunoff_M")

## Temp
total_temp_P = temp_P
total_temp_H = temp_H
total_temp_M = temp_M

annual_temp_P = cache_xarray(compute_annual_generic, total_temp_P, name_hint="annual_temp_P")
annual_temp_H = cache_xarray(compute_annual_generic, total_temp_H, name_hint="annual_temp_H")
annual_temp_M = cache_xarray(compute_annual_generic, total_temp_M, name_hint="annual_temp_M")

### Data visualization: Cartesian grid (but for wintertime)

## REGIONAL

os.makedirs("runoff_ratio", exist_ok=True)

# Little trick to be able to command line pass them
magic = int(sys.argv[1])

region_dic = {1:"West Coast",
              2:"Aleutians",
              3:"Central Interior",
              4:"Northern Slope",
              5:"NE Interior",
              6:"SE Interior",
              7:"Cook Inlet",
              8:"NW Gulf",
              9:"Bristol Bay",
              10:"North Panhandle",
              11:"NE Gulf",
              12:"Central Panhandle",
              13:"South Panhandle"}

total_qrunoff_P = total_qrunoff_P.where(model_grid.OBJECTID.values == magic)
total_qrunoff_H = total_qrunoff_H.where(model_grid.OBJECTID.values == magic)
total_qrunoff_M = total_qrunoff_M.where(model_grid.OBJECTID.values == magic)

# important helper function to get a winter season
def winterizer(ds):
    """Takes a dataset in, returns a version with means for the winter months."""
    djfm = ds.sel(time=ds.time.dt.month.isin([12, 1, 2, 3]))

    def get_season_year(time):
        month = time.dt.month
        year = time.dt.year
        return xr.DataArray(
            year.where(month != 12, year + 1),
            dims='time'
        )
    
    djfm.coords['season_year'] = get_season_year(djfm.time).astype('int32')

    return djfm.groupby('season_year').mean(dim=('lat','lon', 'time'))


# years value we'll use later
years_P = [x.item() for x in winterizer(total_qrunoff_P).season_year.values]
years_H = [x.item() for x in winterizer(total_qrunoff_H).season_year.values]
years_M = [x.item() for x in winterizer(total_qrunoff_M).season_year.values]

# this value is the center of our coordinate grid.
# we define all points relative to these values.
x_origin, y_origin = winterizer(total_precip_P).mean().values, winterizer(total_temp_P).mean().values

## HISTORICAL
# color
precip_ratio_values_P = [(winterizer(total_qrunoff_P).sel(season_year=x).values / winterizer(total_precip_P).sel(season_year=x).values).item()
           for x in years_P]
# y axis 
temp_values_P = [(winterizer(total_temp_P).sel(season_year=x).values).item() - y_origin
           for x in years_P]
# x axis
precip_values_P = [(winterizer(total_precip_P).sel(season_year=x).values).item() - x_origin
           for x in years_P]

## FUTURE HOT
# color
precip_ratio_values_H = [(winterizer(total_qrunoff_H).sel(season_year=x).values / winterizer(total_precip_H).sel(season_year=x).values).item()
           for x in years_H]
# y axis 
temp_values_H = [(winterizer(total_temp_H).sel(season_year=x).values).item() - y_origin
           for x in years_H]
# x axis
precip_values_H = [(winterizer(total_precip_H).sel(season_year=x).values).item() - x_origin
           for x in years_H]

## FUTURE MODERATE
# color
precip_ratio_values_M = [(winterizer(total_qrunoff_M).sel(season_year=x).values / winterizer(total_precip_M).sel(season_year=x).values).item()
           for x in years_M]
# y axis 
temp_values_M = [(winterizer(total_temp_M).sel(season_year=x).values).item() - y_origin
           for x in years_M]
# x axis
precip_values_M = [(winterizer(total_precip_M).sel(season_year=x).values).item() - x_origin
           for x in years_M]

all_heats = np.concatenate([
    precip_ratio_values_P,
    precip_ratio_values_H,
    precip_ratio_values_M
])

vmin, vmax = all_heats.min(), all_heats.max()

print(f"precip ratio_P values: {list(zip(precip_ratio_values_P,years_P))}")
print()
print(f"temp_P values: {list(zip(temp_values_P,years_P))}")
print()
print(f"precip_P values: {list(zip(precip_values_P,years_P))}")
print()
print(f"combined_P coords: {list(zip(precip_values_P, temp_values_P))}")
print()

print(f"precip ratio_H values: {list(zip(precip_ratio_values_H,years_H))}")
print()
print(f"temp_H values: {list(zip(temp_values_H,years_H))}")
print()
print(f"precip_H values: {list(zip(precip_values_H,years_H))}")
print()
print(f"combined_H coords: {list(zip(precip_values_H, temp_values_H))}")
print()

print(f"precip ratio_M values: {list(zip(precip_ratio_values_M,years_M))}")
print()
print(f"temp_M values: {list(zip(temp_values_M,years_M))}")
print()
print(f"precip_M values: {list(zip(precip_values_M,years_M))}")
print()
print(f"combined_M coords: {list(zip(precip_values_M, temp_values_M))}")
print()


# We're defining it manually because using calculated values is extremely error-prone
xmin, xmax = -90, 160
#xmin, xmax = -150, 150
ymin, ymax = -2.5, 7
#ymin, ymax = -14, 14
ticks_x = 10
#ticks_x = 20
ticks_y = 2
#ticks_y = 4

plt.figure(dpi=600)
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin)), aspect='auto')

# axes styling
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Precipitation $(mm/yr)$', size=14, labelpad=-120, x=1.04, rotation=-270)
ax.set_ylabel('Temperature $(Kelvins)$', size=14, labelpad=-21, y=1.02, rotation=0)
#ax.set_xlabel('$x$', size=14, labelpad=-24, x=1.02)
#ax.set_ylabel('$y$', size=14, labelpad=-21, y=1.02, rotation=0)
#plt.text(0.49, 0.49, "$O$", transform=ax.transAxes, ha='center', va='top', fontsize=14)

# grid ticks
ax.xaxis.set_major_locator(MultipleLocator(ticks_x * 5))
ax.xaxis.set_minor_locator(MultipleLocator(ticks_x / 2))
ax.yaxis.set_major_locator(MultipleLocator(ticks_y))
ax.yaxis.set_minor_locator(MultipleLocator(ticks_y / 2))

# TODO: round to int
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{(y + y_origin):.2f}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{(x + x_origin):.2f}'))
#ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))

ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

plt.suptitle(f"Runoff / Precipitation Ratio for Different Precipitation and Temperature Values and Situations (DJFM) ({region_dic[magic]})")

# the easiest way to make plots
sc1 = ax.scatter(precip_values_P, temp_values_P, c=precip_ratio_values_P, label="Past",
                 vmin=vmin, vmax=vmax, cmap="viridis", marker="o")
sc2 = ax.scatter(precip_values_H, temp_values_H, c=precip_ratio_values_H, label="Future (Hot)",
                 vmin=vmin, vmax=vmax, cmap="viridis", marker="s")
sc3 = ax.scatter(precip_values_M, temp_values_M, c=precip_ratio_values_M, label="Future (Moderate)",
                 vmin=vmin, vmax=vmax, cmap="viridis", marker="d")
plt.colorbar(sc1, label="Runoff / Precipitation Ratio", extend="both")
plt.legend()

plt.savefig(f"runoff_ratio/plot-{region_dic[magic]}-DJFM.png")


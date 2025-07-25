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

qflx_evap_paths = ["/data/shared_data/NNA/NNA.4km.hERA5.1989.003/qflx_evap_tot_m/*.nc",
                 "/data/shared_data/NNA/NNA.4km.fPGWh.2033.004/qflx_evap_tot_m/*.nc",
                 "/data/shared_data/NNA/NNA.4km.fPGWm.2033.005/qflx_evap_tot_m/*.nc" ]

snow_P, snow_H, snow_M = unpacker(snow_paths, "SNOW")
rain_P, rain_H, rain_M = unpacker(rain_paths, "RAIN")
temp_P, temp_H, temp_M = unpacker(temp_paths, "TSA")
better_h2osno_P, better_h2osno_H, better_h2osno_M = unpacker(better_h2osno_paths, "H2OSNO")
qrunoff_P, qrunoff_H, qrunoff_M = unpacker(qrunoff_paths, "QRUNOFF")
qflx_evap_P, qflx_evap_H, qflx_evap_M = unpacker(qflx_evap_paths, "QFLX_EVAP_TOT")

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
    
## Snow
total_snow_P = (snow_P) 
total_snow_H = (snow_H) 
total_snow_M = (snow_M)

#total_snow_P = (snow_P) * seconds_per_year
#total_snow_H = (snow_H) * seconds_per_year
#total_snow_M = (snow_M) * seconds_per_year

annual_snow_P = cache_xarray(compute_annual_generic, total_snow_P, name_hint="annual_snow_P")
annual_snow_H = cache_xarray(compute_annual_generic, total_snow_H, name_hint="annual_snow_H")
annual_snow_M = cache_xarray(compute_annual_generic, total_snow_M, name_hint="annual_snow_M")

## Rain
total_rain_P = (rain_P)
total_rain_H = (rain_H)
total_rain_M = (rain_M)

#total_rain_P = (rain_P) * seconds_per_year
#total_rain_H = (rain_H) * seconds_per_year
#total_rain_M = (rain_M) * seconds_per_year

annual_rain_P = cache_xarray(compute_annual_generic, total_rain_P, name_hint="annual_rain_P")
annual_rain_H = cache_xarray(compute_annual_generic, total_rain_H, name_hint="annual_rain_H")
annual_rain_M = cache_xarray(compute_annual_generic, total_rain_M, name_hint="annual_rain_M")

## Precipitation
total_precip_P = (rain_P + snow_P)
total_precip_H = (rain_H + snow_H)
total_precip_M = (rain_M + snow_M)

#total_precip_P = (rain_P + snow_P) * seconds_per_year
#total_precip_H = (rain_H + snow_H) * seconds_per_year
#total_precip_M = (rain_M + snow_M) * seconds_per_year

total_precip_P.name = "PRECIP_P"
total_precip_H.name = "PRECIP_H"
total_precip_M.name = "PRECIP_M"

annual_precip_P = cache_xarray(compute_annual_generic, total_precip_P, name_hint="annual_precip_P")
annual_precip_H = cache_xarray(compute_annual_generic, total_precip_H, name_hint="annual_precip_H")
annual_precip_M = cache_xarray(compute_annual_generic, total_precip_M, name_hint="annual_precip_M")

## Runoff
total_qrunoff_P = qrunoff_P 
total_qrunoff_H = qrunoff_H 
total_qrunoff_M = qrunoff_M 

#total_qrunoff_P = qrunoff_P * seconds_per_year
#total_qrunoff_H = qrunoff_H * seconds_per_year
#total_qrunoff_M = qrunoff_M * seconds_per_year

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

## Evapotranspiration
total_qflx_evap_P = qflx_evap_P
total_qflx_evap_H = qflx_evap_H
total_qflx_evap_M = qflx_evap_M

annual_qflx_evap_P = cache_xarray(compute_annual_generic, total_qflx_evap_P, name_hint="total_qflx_evap_P")
annual_qflx_evap_H = cache_xarray(compute_annual_generic, total_qflx_evap_H, name_hint="total_qflx_evap_H")
annual_qflx_evap_M = cache_xarray(compute_annual_generic, total_qflx_evap_M, name_hint="total_qflx_evap_M")

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

## shadowing ALL the variables with filtered ones
total_qrunoff_P = total_qrunoff_P.where(model_grid.OBJECTID.values == magic)
total_qrunoff_H = total_qrunoff_H.where(model_grid.OBJECTID.values == magic)
total_qrunoff_M = total_qrunoff_M.where(model_grid.OBJECTID.values == magic)

total_temp_P = total_temp_P.where(model_grid.OBJECTID.values == magic)
total_temp_H = total_temp_H.where(model_grid.OBJECTID.values == magic)
total_temp_M = total_temp_M.where(model_grid.OBJECTID.values == magic)

total_precip_P = total_precip_P.where(model_grid.OBJECTID.values == magic)
total_precip_H = total_precip_H.where(model_grid.OBJECTID.values == magic)
total_precip_M = total_precip_M.where(model_grid.OBJECTID.values == magic)

total_rain_P = total_rain_P.where(model_grid.OBJECTID.values == magic)
total_rain_H = total_rain_H.where(model_grid.OBJECTID.values == magic)
total_rain_M = total_rain_M.where(model_grid.OBJECTID.values == magic)

total_snow_P = total_snow_P.where(model_grid.OBJECTID.values == magic)
total_snow_H = total_snow_H.where(model_grid.OBJECTID.values == magic)
total_snow_M = total_snow_M.where(model_grid.OBJECTID.values == magic)

total_evap_P = total_qflx_evap_P.where(model_grid.OBJECTID.values == magic)
total_evap_H = total_qflx_evap_H.where(model_grid.OBJECTID.values == magic)
total_evap_M = total_qflx_evap_M.where(model_grid.OBJECTID.values == magic)

# important helper function to get a winter season
def winterizer(ds, dataset_name, scale_by_time=True):
    """returns winter totals (dec–mar) per season_year; optionally scaled by number of seconds"""
    djfm = ds.sel(time=ds.time.dt.month.isin([12, 1, 2, 3]))

    def get_season_year(time):
        month = time.dt.month
        year = time.dt.year
        return xr.DataArray(year.where(month != 12, year + 1), dims='time')

    season_years = get_season_year(djfm.time).astype('int32')
    djfm.coords['season_year'] = season_years

    def get_seconds_per_winter(season_years):
        years = np.unique(season_years.values)
        starts = pd.to_datetime([f"{y-1}-12-01" for y in years])
        ends = pd.to_datetime([f"{y}-04-01" for y in years])
        durations = (ends - starts).days * 86400
        return xr.DataArray(durations, dims='season_year', coords={'season_year': years})

    seconds_per_year = get_seconds_per_winter(season_years)

    def groupby_reducer(ds):
        grouped = ds.groupby('season_year').mean(dim=('lat', 'lon', 'time'))
        grouped = grouped * seconds_per_year if scale_by_time else grouped
        grouped.name = dataset_name
        return grouped

    return cache_xarray(groupby_reducer, djfm, name_hint=dataset_name)

# years value we'll use later
years_P = [x.item() for x in winterizer(total_qrunoff_P, f"winterized_qrunoff_P_{magic}").season_year.values]
years_H = [x.item() for x in winterizer(total_qrunoff_H, f"winterized_qrunoff_H_{magic}").season_year.values]
years_M = [x.item() for x in winterizer(total_qrunoff_M, f"winterized_qrunoff_M_{magic}").season_year.values]

# this value is the center of our coordinate grid.
# we define all points relative to these values.
x_origin = winterizer(total_precip_P, f"winterized_precip_P_{magic}").mean().values
y_origin = winterizer(total_temp_P, f"winterized_temp_P_{magic}", scale_by_time=False).mean().values

print(f"x-origin: {x_origin}, y-origin: {y_origin}")

## HISTORICAL
# color
precip_ratio_values_P = [(winterizer(total_qrunoff_P, f"winterized_qrunoff_P_{magic}").sel(season_year=x) / winterizer(total_precip_P, f"winterized_precip_P_{magic}").sel(season_year=x)).item()
                         for x in years_P]
# y axis 
temp_values_P = [(winterizer(total_temp_P, f"winterized_temp_P_{magic}", scale_by_time=False).sel(season_year=x)).item() - y_origin
                 for x in years_P]
# x axis
precip_values_P = [(winterizer(total_precip_P, f"winterized_precip_P_{magic}").sel(season_year=x)).item() - x_origin
                   for x in years_P]
print("historical (P) done")

## HISTORICAL-DEBUG
# color
precip_ratio_values_debug_P = winterizer(total_qrunoff_P, f"winterized_qrunoff_P_{magic}").values / winterizer(total_precip_P, f"winterized_precip_P_{magic}").values
print(f"sanity test: is debug and real equal (precipitation ? {precip_ratio_values_P == precip_ratio_values_debug_P}")

# runoff
runoff_values_debug_P = [(winterizer(total_qrunoff_P, f"winterized_qrunoff_P_{magic}").sel(season_year=x)).item()
                         for x in years_P]
# y axis 
temp_values_debug_P = [(winterizer(total_temp_P, f"winterized_temp_P_{magic}", scale_by_time=False).sel(season_year=x)).item()
                       for x in years_P]
# x axis
precip_values_debug_P = [(winterizer(total_precip_P, f"winterized_precip_P_{magic}").sel(season_year=x)).item()
                         for x in years_P]
# snow (sanity check)
snow_values_debug_P = [(winterizer(total_snow_P, f"winterized_snow_P_{magic}").sel(season_year=x)).item()
                         for x in years_P]
# rain (sanity check)
rain_values_debug_P = [(winterizer(total_rain_P, f"winterized_rain_P_{magic}").sel(season_year=x)).item()
                         for x in years_P]
# runoff (sanity check)
runoff_values_debug_P = [(winterizer(total_runoff_P, f"winterized_runoff_P_{magic}").sel(season_year=x))
                         for x in years_P]
# evap (sanity check)
evap_values_debug_P = [(winterizer(total_evap_P, f"winterized_evap_P_{magic}").sel(season_year=x))
                         for x in years_P]
print("historical (P-DEBUG) done")

## FUTURE HOT
# color
precip_ratio_values_H = [(winterizer(total_qrunoff_H, f"winterized_qrunoff_H_{magic}").sel(season_year=x).values / winterizer(total_precip_H, f"winterized_precip_H_{magic}").sel(season_year=x).values).item()
                         for x in years_H]

# y axis 
temp_values_H = [(winterizer(total_temp_H, f"winterized_temp_H_{magic}", scale_by_time=False).sel(season_year=x).values).item() - y_origin
                 for x in years_H]
# x axis
precip_values_H = [(winterizer(total_precip_H, f"winterized_precip_H_{magic}").sel(season_year=x).values).item() - x_origin
                   for x in years_H]
print("future (H) done")

## FUTURE HOT-DEBUG
# color
precip_ratio_values_debug_H = winterizer(total_qrunoff_H, f"winterized_qrunoff_H_{magic}").values / winterizer(total_precip_H, f"winterized_precip_H_{magic}").values
print(f"sanity test: is debug and real equal (precipitation H)? {precip_ratio_values_H == precip_ratio_values_debug_H}")

# runoff
runoff_values_debug_H = [(winterizer(total_qrunoff_H, f"winterized_qrunoff_H_{magic}").sel(season_year=x).values).item()
                         for x in years_H]
# y axis 
temp_values_debug_H = [(winterizer(total_temp_H, f"winterized_temp_H_{magic}", scale_by_time=False).sel(season_year=x).values).item()
                       for x in years_H]
# x axis
precip_values_debug_H = [(winterizer(total_precip_H, f"winterized_precip_H_{magic}").sel(season_year=x).values).item()
                         for x in years_H]
# snow (sanity check)
snow_values_debug_H = [(winterizer(total_snow_H, f"winterized_snow_H_{magic}").sel(season_year=x)).item()
                         for x in years_H]
# rain (sanity check)
rain_values_debug_H = [(winterizer(total_rain_H, f"winterized_rain_H_{magic}").sel(season_year=x)).item()
                         for x in years_H]
# runoff (sanity check)
runoff_values_debug_H = [(winterizer(total_runoff_H, f"winterized_runoff_H_{magic}").sel(season_year=x))
                         for x in years_H]
# evap (sanity check)
evap_values_debug_H = [(winterizer(total_evap_H, f"winterized_evap_H_{magic}").sel(season_year=x))
                         for x in years_H]
print("future (H-DEBUG) done")

## FUTURE MODERATE
# color
precip_ratio_values_M = [(winterizer(total_qrunoff_M, f"winterized_qrunoff_M_{magic}").sel(season_year=x).values / winterizer(total_precip_M, f"winterized_precip_M_{magic}").sel(season_year=x).values).item()
           for x in years_M]
# y axis 
temp_values_M = [(winterizer(total_temp_M, f"winterized_temp_M_{magic}", scale_by_time=False).sel(season_year=x).values).item() - y_origin
           for x in years_M]
# x axis
precip_values_M = [(winterizer(total_precip_M, f"winterized_precip_M_{magic}").sel(season_year=x).values).item() - x_origin
           for x in years_M]
print("future (M) done")

## FUTURE MODERATE-DEBUG
# color
precip_ratio_values_debug_M = winterizer(total_qrunoff_M, f"winterized_qrunoff_M_{magic}").values / winterizer(total_precip_M, f"winterized_precip_M_{magic}").values
print(f"sanity test: is debug and real equal (precipitation M)? {precip_ratio_values_M == precip_ratio_values_debug_M}")

# runoff
runoff_values_debug_M = [(winterizer(total_qrunoff_M, f"winterized_qrunoff_M_{magic}").sel(season_year=x).values).item()
                         for x in years_M]
# y axis 
temp_values_debug_M = [(winterizer(total_temp_M, f"winterized_temp_M_{magic}", scale_by_time=False).sel(season_year=x).values).item()
                       for x in years_M]
# x axis
precip_values_debug_M = [(winterizer(total_precip_M, f"winterized_precip_M_{magic}").sel(season_year=x).values).item()
                         for x in years_M]
# snow (sanity check)
snow_values_debug_M = [(winterizer(total_snow_M, f"winterized_snow_M_{magic}").sel(season_year=x)).item()
                         for x in years_M]
# rain (sanity check)
rain_values_debug_M = [(winterizer(total_rain_M, f"winterized_rain_M_{magic}").sel(season_year=x)).item()
                         for x in years_M]
# runoff (sanity check)
runoff_values_debug_M = [(winterizer(total_runoff_M, f"winterized_runoff_M_{magic}").sel(season_year=x))
                         for x in years_M]
# evap (sanity check)
evap_values_debug_M = [(winterizer(total_evap_M, f"winterized_evap_M_{magic}").sel(season_year=x))
                         for x in years_M]
print("future (M-DEBUG) done")

all_precip_ratio = np.concatenate([
    precip_ratio_values_P,
    precip_ratio_values_H,
    precip_ratio_values_M
])

all_precips = np.concatenate([
    precip_values_P,
    precip_values_H,
    precip_values_M
])

all_temps = np.concatenate([
    temp_values_P,
    temp_values_H,
    temp_values_M
])

## polyline calculation
slope, intercept = np.polyfit(all_precips, all_temps, 1)
x_line = np.linspace(all_precips.min(), all_precips.max(), 100)
y_line = slope * x_line + intercept

vmin, vmax = all_precip_ratio.min(), all_precip_ratio.max()

## Debugging symbols
log_file = open(f"runoff_ratio/plot-{region_dic[magic]}-DJFM.log.txt", 'w')

log_file.write(f"######## Data log for {region_dic[magic]}\n")

log_file.write("#### HISTORICAL\n")
log_file.write(f"precip ratio_P values: {list(zip(precip_ratio_values_debug_P,years_P))}\n")
log_file.write(f"temp_P values: {list(zip(temp_values_debug_P,years_P))}\n")
log_file.write(f"precip_P values: {list(zip(precip_values_debug_P,years_P))}\n")

log_file.write("#### FUTURE (HOT)\n")
log_file.write(f"precip ratio_H values: {list(zip(precip_ratio_values_debug_H,years_H))}\n")
log_file.write(f"temp_H values: {list(zip(temp_values_debug_H,years_H))}\n")
log_file.write(f"precip_H values: {list(zip(precip_values_debug_H,years_H))}\n")

log_file.write("#### FUTURE (MODERATE)\n")
log_file.write(f"precip ratio_M values: {list(zip(precip_ratio_values_debug_M,years_M))}\n")
log_file.write(f"temp_M values: {list(zip(temp_values_debug_M,years_M))}\n")
log_file.write(f"precip_M values: {list(zip(precip_values_debug_M,years_M))}\n")

log_file.close()

## netCDF exporting
## expand this if you want more of them in the files
variables_historical = {
    "temp_P": temp_values_debug_P,
    "snow_P": snow_values_debug_P,
    "rain_P": rain_values_debug_P,
    "precip_P": precip_values_debug_P,
    "runoff_P": runoff_values_debug_P,
    "evap_P": evap_values_debug_P,
}
variables_future = {
    "temp_H": temp_values_debug_H,
    "snow_H": snow_values_debug_H,
    "rain_H": rain_values_debug_H,
    "precip_H": precip_values_debug_H,
    "runoff_H": runoff_values_debug_H,
    "evap_H": evap_values_debug_H,
    "temp_M": temp_values_debug_M,
    "snow_M": snow_values_debug_M,
    "rain_M": rain_values_debug_M,
    "precip_M": precip_values_debug_M,
    "runoff_M": runoff_values_debug_M,
    "evap_M": evap_values_debug_M,
}
ds_historical = xr.Dataset(
    {k: ("year", v) for k, v in variables_historical.items()},
    coords={"year": years_P},
)
ds_future = xr.Dataset(
    {k: ("year", v) for k, v in variables_future.items()},
    coords={"year": years_H},
)
historical_file_name = f"runoff_ratio/{region_dic[magic]}_djfm_hist_data.nc"
future_file_name = f"runoff_ratio/{region_dic[magic]}_djfm_future_data.nc"
ds_historical.to_netcdf(historical_file_name)
ds_future.to_netcdf(future_file_name)


# stupid hack but kinda works actually
xmin, xmax = all_precips.min() - 5, all_precips.max() + 5
#xmin, xmax = -150, 150
ymin, ymax = all_temps.min() - 2, all_temps.max() + 2
#ymin, ymax = -14, 14
ticks_x = 20
#ticks_x = 20
ticks_y = 2
#ticks_y = 4

fig, ax = plt.subplots(figsize=(12, 8), dpi=600, constrained_layout=True)
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
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y + y_origin)}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x + x_origin)}'))
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
# add our polyline now that everything else is plotted
ax.plot(x_line, y_line, 'red')

plt.legend()

plt.savefig(f"runoff_ratio/plot-{region_dic[magic]}-DJFM.png")

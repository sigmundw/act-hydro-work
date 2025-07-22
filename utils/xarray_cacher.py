import glob
import os
import hashlib
import io
import tempfile
import logging

import numpy as np
import xarray as xr
import pandas as pd
import zstandard as zstd

def unpacker(paths, var):
    """Takes a list of datasets, opens all files and returns a list of datasets for each."""
    storage = []
    for path in paths:
        storage += [xr.open_mfdataset(sorted(glob.glob(path)), 
                                      combine="by_coords",
                                      chunks={'time': 100})[var]]
    return storage

def unpacker_chunkless(paths, var):
    """Takes a list of datasets, opens all files and returns a list of datasets for each."""
    storage = []
    for path in paths:
        storage += [xr.open_mfdataset(sorted(glob.glob(path)), 
                                      combine="by_coords")[var]]
    return storage

def cache_xarray(fn, var, kwargs=None, name_hint="", cache_dir="./cache"):
    """
    Unlike a bare method call, this function is capable of returning a cache hit if one exists,
    which can massively save on execution time. Only the exact same invocation will result in a hit.

    Call like:
    def compute_annual_q(ds):
        return ds.groupby("time.year").mean(dim="time", skipna=True)

    annual_qrunoff_P = cache_xarray(compute_annual_q, total_qrunoff_P, name_hint="annual_qrunoff_P")

    Alternatively, pass fn as None to skip applying any function at all.
    """
    if kwargs is None:
        kwargs = {} # more of a sanity check than anything
        
    os.makedirs(cache_dir, exist_ok=True)

    # TODO: add a command line arg to cache_xarray that lets you switch the value
    # this disables all output, turn this way down to make it noisier. 
    
    #logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=100)
    logger = logging.getLogger(__name__)

    logger.debug("Creating hash...")

    def hash_var(var):
        h = hashlib.sha256()
        h.update(str(var.name).encode())
        h.update(str(var.shape).encode())
        h.update(str(var.dtype).encode())
        h.update(str(var.coords).encode())
        h.update(str(var.attrs).encode())
        return h.hexdigest()

    def hash_string(s):
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
    
    hash_key_arg = "".join(hash_var(var))
    cache_path = os.path.join(cache_dir, f"{name_hint}_{hash_key_arg}.nc")

    logger.debug(f"Hash: {name_hint}_{hash_key_arg}")
    
    if os.path.exists(cache_path + '.zstd'):
        logger.debug("Cache hit!")
        dctx = zstd.ZstdDecompressor()
        with open(cache_path + '.zstd', 'rb') as file_in:
            with dctx.stream_reader(file_in) as reader:
                buffer = io.BytesIO(reader.read())
        ds = xr.load_dataset(buffer)
        if len(ds.data_vars) == 1: # if only one variable, return it
            item = next(iter(ds.data_vars.values()))
            logger.debug(f"{name_hint}'s cache load: is type {type(item)} post-load with var {item.name}")
            logger.info(f"Successfully loaded {name_hint}")
            return item
        else:
            logger.debug(f"{name_hint}'s cache load: is type {type(ds)} post-load with NO var detected")
            logger.info(f"Successfully loaded {name_hint}")
            return ds # if it's a Dataset, not a DataArray
    else:
        logger.debug("No cache, calculating...")
        if fn != None:
            result = fn(var, **kwargs)
        else:
            result = var
        # TODO: should there be Dataset support? It could be added, but would add complexity
        if not isinstance(result, (xr.DataArray)):
            raise TypeError(f"Refusing to cache non-DataArray object: got {type(result)}")
        else:
            if result.name is None:
                raise ValueError("Resulting DataArray has no name, can't serialize cleanly.")
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file: 
            tmp_file_name = tmp_file.name
            # we use a tempfile here because .to_netcdf() gets angry if it can't control closing the file
            result.to_dataset(name=var.name).to_netcdf(tmp_file, engine="netcdf4")
        with open(tmp_file_name, 'rb') as file:
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(file.read())
            with open(cache_path + '.zstd', 'wb') as file_out:
                logger.debug(f"{name_hint}'s writing: is type {type(result)} with var {var.name}")
                file_out.write(compressed)
        os.remove(tmp_file_name) # no longer needed
        logger.info(f"Successfully created cache for {name_hint}")
        return result
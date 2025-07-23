# Single Variable Code
These notebooks all test one particular variable of the climate model data, and should produce some nice starting points if you're looking to investigate a different variable. You can run any of them by entering the environment described in the [main README](../README.md) and then clicking run in Jupyter.

Currently, this code uses the old `cache_xarray` definitions instead of the newer `zeitcache` ones. You'll want to consider overriding the cache locations if you'd like so that you get better hits (as other code in other directories will make many of the same variables as in this one).

# Snow Water Equivalent Calculations
These files are used for calculating the amount of water that gets melted off over the course of a year by determining how much snow fell in a year, and dividing that by the maximum SWE on April 1st. The notebook shows an example of this process, but it's far, far too slow. You should run one of the Python files instead, as they'll be able to calculate it faster.

As the name implies, `snow_and_rain_math.py` is focused on also including frozen rain in those calculations (any rain that falls during the snow season is counted as "frozen rain" due to lack of a better method). `snow_math.py` just uses the snow data. You will get strange results depending on the dataset that you are using. 

Unlike the Runoff Ratio code, there is no need for Parallel here, but it is strongly recommended that you tweak the values in `snow_and_rain_math.py` and `snow_math.py` before executing this code, otherwise you may overwhelm your computer with the Dask cluster's computational needs. This code was originally written for supercomputers; on a home computer, you'll need to tune these numbers down significantly and be quite patient.

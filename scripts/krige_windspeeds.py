import os
from tqdm import tqdm, trange
# import sys
from glob import glob
# Better debugging printing
from icecream import ic
# Working with data
import xarray as xr
import numpy as np
import pandas as pd
import regionmask
import scipy.stats as stats
# Parallel processing
from joblib import Parallel, delayed, parallel_config
import gstools as gs
import matplotlib.pyplot as plt

def load_states(path):
    # Get all station files
    state_files = [
        f for f in glob(path+'*.nc') if
        f.find('Alaska') == -1 and f.find('Hawaii') == -1
    ]
    l48 = (
        xr.open_mfdataset(state_files, combine='nested', concat_dim='state')
        .sel(time=slice("1950-01-01", None))
        .where(lambda x: (x['windspeeds'] >= 0).compute(), drop=True)
        .resample(time='1D').mean()
        .chunk("auto")
    )
    return l48
    
def krige_states(state_path, temis_precision="0500deg"):
    # Load in station data 
    print('Loading station data...')
    l48 = load_states(state_path)
    # Load in temis data
    print('Loading TEMIS elevations data...')
    t_file = glob(f'../data/TEMIS/*{temis_precision}*.nc')[0]
    temis = (
        xr.load_dataset(t_file)
        .isel(nbounds=1)
    )
    # Mask by states
    print('Masking TEMIS data...')
    state_mask = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    # Hawaii and Alaska are not included in the mask
    good_keys = [
        k for k in state_mask.regions.keys() 
        if k not in state_mask.map_keys(['Hawaii', 'Alaska'])
    ]
    temis_state = temis.where(
        state_mask.mask(temis['longitude'], temis['latitude']).isin(good_keys), 
        drop=True
    )
    
    # Generate grid for prediction
    print('Generating grid...')
    lons, lats = np.meshgrid(
        temis_state['longitude'].values.flatten(),
        temis_state['latitude'].values.flatten()
    )
    
    # Generate dataset to hold values
    ds = xr.Dataset(
        # data_vars=dict(
        #     temperature=(["x", "y", "time"], temperature),
        # ),
        coords=dict(
            lon=(["x", "y"], lons),
            lat=(["x", "y"], lats),
            time=(["time"], pd.date_range("1950-01-01", "2022-12-31", freq="M")),
        ),
        attrs=dict(description=f'Kriged windspeeds at {temis} resolution'),
    )
    elev_shape = temis_state['elevation'].shape
    
    # Start the kriging process
    fit_model = gs.Stable(latlon=True, var=0.921, len_scale=1.14, nugget=0.639, geo_scale=57.3, alpha=0.666)
    # Loop over each time period
    start_year = 1950
    fields = []
    stds = []
    for y in trange(2022-start_year+1, desc='Kriging data [years]'):
        for m in trange(12, desc='Months', leave=False):
            month = m+1
            # Select data
            month = f'{start_year+y}-{month:02d}'
            # end = f'{start_year}-{month:02d}-31'
            l48_stat = (
                l48.sel(time=month)
                .load()
                .where(lambda x: x['windspeeds'].notnull(), drop=True)
            )
            l48_mon = l48_stat.mean('time')
            # Generate errors
            n = l48_stat['station_id'].size
            errs = []
            for s in range(n):
                # Station data
                station = l48_stat.isel(station_id=s)
                # Filter out bad windspeeds
                wind = station['windspeeds'].values
                # Cant use 0 for gamma distribution estimations
                good_wind = wind[wind > 0]
                if len(good_wind) <= 1:
                    # If no non-zero winds or just one good observation, cannot determine error
                    errs.append(np.nan)
                else:
                    ag,bg,cg = stats.gamma.fit(good_wind, floc=0)
                    errs.append(stats.gamma.std(ag,bg,cg))
                    
            # For sites with one observation, set error to be mean of all other errors
            errs = np.array(errs)
            errs[np.isnan(errs)] = np.nanmean(errs)
            # l48_mon['gamma_err'] = xr.DataArray(errs, dims='station_id', coords={'station_id': l48_mon['station_id']})
            # l48_mon['gamma_err'].attrs['units'] = 'm/s'
            # Get temis data at station locations
            temis_station = temis.sel(
                latitude=l48_mon['latitude'], 
                longitude=l48_mon['longitude'], 
                method='nearest'
            )
            # Get holdout stations
            # holdout =  np.random.choice(n, int(np.floor(len(n*0.1))), replace=False)
            # Take indexes *not* in holdout stations
            # train = np.setdiff1d(np.arange(n), holdout)
            train_drift = [
                temis_station['elevation'].values.flatten(),
                temis_station['elevation_stddev'].values.flatten()
            ]
            krig = gs.krige.ExtDrift(
                model=fit_model,
                cond_pos=[
                    l48_mon['longitude'].values.flatten(),
                    l48_mon['latitude'].values.flatten()
                ],
                cond_val=l48_mon['windspeeds'].mean('state').values.flatten(),
                ext_drift = train_drift,
                exact=False,
                cond_err=errs
            )
            all_drift = [
                temis_state['elevation'].values.flatten(),
                temis_state['elevation_stddev'].values.flatten()
            ]
            field, variance = krig((lons, lats), ext_drift=all_drift)
            # Reshape to 2D grid
            k_field = field.reshape(elev_shape)
            std_field = np.sqrt(variance).reshape(elev_shape)
            # Write to file
            label_year = start_year+y
            k_field.dump(f'../data/kriged_HadISD/{label_year}-{month}-k_field.npy')
            std_field.dump(f'../data/kriged_HadISD/{label_year}-{month}-std_field.npy')
            # Store data
            fields.append(k_field)
            stds.append(std_field)
        # Increment year
        # start_year += 1
    # Convert to numpy arrays and save
    print('Saving data...')
    fields = np.array(fields)
    stds = np.array(stds)
    fields.dump('fields.npy')
    stds.dump('stds.npy')
    # Add data variables to dataset
    ds['windspeeds'] = xr.DataArray(np.array(fields), dims=['time', 'x', 'y'])
    ds['windspeeds'].attrs['units'] = 'm/s'
    ds['windspeeds'].attrs['description'] = 'Kriged windspeeds'
    ds['windspeeds_std'] = xr.DataArray(np.array(stds), dims=['time', 'x', 'y'])
    ds['windspeeds_std'].attrs['units'] = 'm/s'
    ds['windspeeds_std'].attrs['description'] = 'Kriging standard deviation'
    # Save dataset
    print('Saving dataset...')
    ds.to_netcdf(f'../data/kriged_windspeeds_{temis_precision}.nc', engine='netcdf4')


def viz_kriged(file):
    ds = xr.load_dataset(file)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.errorbar(ds.time, ds.windspeeds.mean({'lat', 'lon'}), yerr=ds.windspeeds_std.mean({'lat', 'lon'}))
    ax.set_xlabel('Time')
    ax.set_ylabel('Windspeed [m/s]')
    ax.set_title('Mean Lower 48 Windspeeds')
    plt.show()
    
    
    
if __name__ == "__main__":
    krige_states('../data/HadISD/us-states/')
    
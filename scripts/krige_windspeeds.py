import os
from tqdm import tqdm, trange
# import sys
from glob import glob
# Better debugging printing
# from icecream import ic
# Working with data
import xarray as xr
import numpy as np
import pandas as pd
import regionmask
import scipy.stats as stats
# Parallel processing
# from joblib import Parallel, delayed, parallel_config
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
        .sel(time=slice("2021-01-01", None))
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
            time=(["time"], pd.date_range("2021-01-01", "2022-12-31", freq="M")),
        ),
        attrs=dict(description=f'Kriged windspeeds at {temis} resolution'),
    )
    elev_shape = temis_state['elevation'].shape
    
   # Use spherical model (from Joyner et al. 2015)
    # base_model = gs.Spherical(latlon=True, geo_scale=gs.tools.KM_SCALE)
    # Loop over each time period
    start_year = 2021
    fields = []
    stds = []
    for y in trange(2022-start_year+1, desc='Kriging data [years]'):
        current_year = start_year+y
        # This is the long step
        year_data = (
            l48
            .sel(time=slice(f'{current_year}-01-01', f'{current_year}-12-31'))
            .resample(time='1D').mean()
            .load()
        )
        for m in trange(12, desc='Months', leave=False):
            month = m+1
            time_label = str(current_year)+'-'+str(month)
            # Select data
            month = f'{current_year}-{month:02d}'
            l48_stat = (
                year_data.sel(time=month)
                .where(lambda x: x['windspeeds'] > 0, drop=True)
                .where(lambda x: x['windspeeds'] < 100, drop=True)
            )
            l48_mon = l48_stat.mean('time')
            # Generate errors
            n = l48_stat['station_id'].size
            errs = []
            for s in range(n):
                # Station data (no need to take mean by state, filter will remove nans)
                station = l48_stat.isel(station_id=s)
                # Filter out bad windspeeds
                wind = station['windspeeds'].mean('state').values
                # Cant use 0 for gamma distribution estimations
                good_wind = wind[(wind > 0) & (wind < 100)]
                # Want 20+ observations per month
                if len(good_wind) < 20:
                    # If no non-zero winds or just one good observation, cannot determine error
                    errs.append(-999)
                else:
                    # If all measurements are the same this will fail
                    ag,bg,cg = stats.gamma.fit(good_wind, floc=0)
                    errs.append(stats.gamma.std(ag,bg,cg))
            errs = np.array(errs)
            # For stations with bad data, get their index
            good_stations = np.where(errs != -999)
            # Data 
            cond_pos = [
                l48_mon['longitude'].values.flatten()[good_stations],
                l48_mon['latitude'].values.flatten()[good_stations]
            ]
            cond_val = l48_mon['windspeeds'].mean('state').values.flatten()[good_stations]
            # Determine variogram model
            models = {
                "Gaussian": gs.Gaussian,
                "Exponential": gs.Exponential,
                "Matern": gs.Matern,
                "Stable": gs.Stable,
                "Rational": gs.Rational,
                # "Circular": gs.Circular, # no good in 3D
                "Spherical": gs.Spherical,
                "SuperSpherical": gs.SuperSpherical,
                "JBessel": gs.JBessel,
            }
            # Max bin size is 5 degrees, group by TEMIS elevation
            # bin_size = float(temis_precision[:4])/1000
            # bins = np.arange(0, 5, bin_size) # max window of 5 degrees
            # Get bin number from Sturges Rule, max distance of 500 km
            bins = gs.variogram.standard_bins(cond_pos, max_dist = 500, latlon=True, geo_scale=gs.tools.KM_SCALE)
            # bins = gs.variogram.standard_bins(variogram_pos, bin_no=499, max_dist = 5, latlon=True, geo_scale=gs.tools.DEGREE_SCALE)
            bin_center, gamma = gs.vario_estimate(
                cond_pos, cond_val, bins,
                latlon=True, geo_scale=gs.tools.KM_SCALE
            )
            # fit all models to the estimated variogram
            best_r2 = 0
            best_model = None
            for model in models:
                fit_model = models[model](
                    latlon=True, geo_scale=gs.tools.KM_SCALE
                )
                try:
                    # nugget = False because we have measurement
                    para, pcov, r2 = fit_model.fit_variogram(
                        bin_center, gamma, return_r2=True
                    )
                    if r2 > best_r2:
                        best_model = fit_model
                        best_r2 = r2                    
                except:
                    continue

            # para, pcov, r2 = base_model.fit_variogram(
            #     bin_center, gamma, return_r2=True
            # )
            
            ax = best_model.plot(x_max=500)
            ax.scatter(bin_center, gamma)
            ax.set_xlabel('Distance [km]')
            ax.set_ylabel('Semivariance')
            ax.set_title(f'{time_label} variogram (R$^2$={r2:.2f})')
            plt.savefig(f'../plots/kriged-variograms/'+time_label+'-variogram.png')  

            temis_station = temis.sel(
                latitude=l48_mon['latitude'], 
                longitude=l48_mon['longitude'], 
                method='nearest'
            )
            train_drift = [
                temis_station['elevation'].values.flatten()[good_stations],
                temis_station['elevation_stddev'].values.flatten()[good_stations]
            ]
            krig = gs.krige.ExtDrift(
                model=best_model,
                cond_pos=cond_pos,
                cond_val=cond_val,
                ext_drift = train_drift,
                exact=False,
                cond_err=errs[good_stations]
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
            k_field.dump(f'../data/kriged_HadISD/'+time_label+'-k_field.npy')
            std_field.dump(f'../data/kriged_HadISD/'+time_label+'-std_field.npy')
            # Store data
            fields.append(k_field)
            stds.append(std_field)
            
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


def combine_fields(path, temis_precision="0500deg"):
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
    
    # Load fields
    k_files = sorted(glob(f'{path}/*k_field.npy'))
    std_files = sorted(glob(f'{path}/*std_field.npy'))
    fields = np.load(k_files, allow_pickle=True)
    stds = np.load(std_files, allow_pickle=True)
    
    # Generate dataset to hold values
    ds = xr.Dataset(
        data_vars=dict(
            windspeeds=(['time', 'x', 'y'], fields),
            windspeeds_std=(['time', 'x', 'y'], stds),
        ),
        coords=dict(
            lon=(["x", "y"], lons),
            lat=(["x", "y"], lats),
            time=(["time"], pd.date_range("2021-01-01", "2022-12-31", freq="M")),
        ),
        attrs=dict(description=f'Kriged windspeeds at {temis} resolution'),
    )
    # Add data variables to dataset
    ds['windspeeds'].attrs['units'] = 'm/s'
    ds['windspeeds'].attrs['description'] = 'Kriged windspeeds'
    ds['windspeeds_std'].attrs['units'] = 'm/s'
    ds['windspeeds_std'].attrs['description'] = 'Kriging standard deviation'
    print('Saving dataset...')
    ds.to_netcdf(f'../data/kriged_windspeeds_{temis_precision}.nc', engine='netcdf4')
    
    
    
if __name__ == "__main__":
    krige_states('../data/HadISD/us-states/')
    
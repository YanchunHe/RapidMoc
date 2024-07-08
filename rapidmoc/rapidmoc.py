"""
Module containing main routines to execute RapidMoc

"""


import argparse
import configparser
import copy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import utils
import sections
import transports
import observations
import plotdiag



def get_args():
    """   Get arguments from command line.  """
    parser = argparse.ArgumentParser(
        description='Calculate RAPID AMOC diagnostics using ocean model data')
    parser.add_argument(
        'config_file', type=str, help='Path to configuration file.')
    parser.add_argument(
        'tfile', type=str, help='Path for netcdf file(s) containing temperature data.')
    parser.add_argument(
        'sfile', type=str, help='Path for netcdf file(s) containing salinity data.')
    parser.add_argument(
        'taufile', type=str, help='Path for netcdf file(s) containing zonal wind stress data.')
    parser.add_argument(
        'vfile', type=str, help='Path for netcdf file(s) containing meridional velocity data.')
    parser.add_argument(
        'vflxfile', type=str, help='Path for netcdf file(s) containing mass flux in y direction data.')
    parser.add_argument(
        'dpfile',type=str, help = 'Path for netctdf file containing pressure layer')
    parser.add_argument(
        '--name', help='Name used in output files. Overrides value in config file.', default=None)
    parser.add_argument(
        '--outdir', help='Path used for output data. Overrides value in config file.', default=None)
    args = parser.parse_args()
    
    return args



def get_config(args):
    """ Return configuration options as <ConfigParser> object. """
    config = configparser.ConfigParser()
    config.read(args.config_file)

    return config


def get_config_opt(config, section, option):
    """ Return option if exists, else None """
    if config.has_option(section, option):
        return config.get(section, option)
    else:
        return None


def call_plotdiag(config, trans):
    """ Call plotting routines to compare against RAPID observations """ 

    # Initialize observations
    obs_fc, obs_oht, obs_vol, obs_sf = None, None, None, None

    # Get observation file paths
    time_avg = get_config_opt(config, 'observations', 'time_avg')
    obs_sf_f = get_config_opt(config, 'observations', 'streamfunctions')
    #obs_fc_f = get_config_opt(config, 'observations', 'florida_current')
    obs_vol_f = get_config_opt(config, 'observations', 'volume_transports')
    #obs_oht_f = get_config_opt(config, 'observations', 'heat_transports')
    
    # Load observations, if specified
    #if obs_oht_f is not None:
       # obs_oht = observations.HeatTransportObs(obs_oht_f, time_avg=time_avg)

    #if obs_fc_f is not None:
        #obs_fc = observations.FloridaCurrentObs(obs_fc_f, time_avg=time_avg)

    if obs_sf_f is not None:
        obs_sf = observations.StreamFunctionObs(obs_sf_f, time_avg=time_avg)

    if obs_vol_f is not None:
        obs_vol = observations.VolumeTransportObs(obs_vol_f, time_avg=time_avg)
 
    # Call plot routines
    outdir = config.get('output', 'outdir')
    date_format = config.get('output', 'date_format')
    name = config.get('output', 'name')
    plotdiag.plot_diagnostics(trans, name=name, outdir=outdir, date_format=date_format,
                              obs_vol=obs_vol, obs_fc=obs_fc, obs_oht=obs_oht, obs_sf=obs_sf)





def main():
    """ Parse options and run RapidMoc. """
    args = get_args()
    config = get_config(args)
    

    # Update name in config file
    if args.name is not None:
        config.set('output', 'name', args.name)

    # Update outdir in config file
    if args.outdir is not None:
        config.set('output', 'outdir', args.outdir)

    # Read data
    
    t = sections.ZonalSections(args.tfile, config, 'temperature')
    
    s = sections.ZonalSections(args.sfile, config, 'salinity')
    tau = sections.ZonalSections(args.taufile, config, 'taux')
    v = sections.ZonalSections(args.vfile, config, 'meridional_velocity')
    ## adding data to claculate the transport with mass flux 
    vflx = sections.ZonalSections(args.vflxfile, config, 'mass_flux_y_direction')
    dp = sections.ZonalSections(args.dpfile, config, 'pressure_layer')
    # Interpolate T & S data onto v-grid
    t_on_v = sections.interpolate(t,v)
    s_on_v = sections.interpolate(s,v)
    
    t_on_vflx = sections.interpolate(t,vflx)
    s_on_vflx = sections.interpolate(s,vflx)
    print('t_on_vflx', t_on_vflx.data.shape)
    
    # Return integrated transports on RAPID section as netcdf object
    #trans = transports_test.calc_transports_from_sections(
       # config, vflx, v,tau, t, s, t_on_v,s_on_v)
    #minind,maxind=utils.get_indrange(vals,minval,maxval)
    
    
    trans = transports.calc_transports_from_sections(config, vflx,v, tau, t_on_vflx, s_on_vflx, t_on_v,s_on_v)
    ## Plot diagnostics
    if config.getboolean('output','plot'):
        call_plotdiag(config, trans)
        
    # Write data
    print('SAVING: %s' % trans.filepath())
    

#def plot_transports(trans):    # Supposons que trans soit un NetCDF Dataset et que vous souhaitiez tracer une variable appelée 'transport_data'
 #   try:
  #      file=trans.filepath()
   #     d=xr.open_dataset(file)
    #    transport_data = d['sf_ek'][:]
   # except KeyError:
    #    print("Variable 'transport_data' not found in the dataset.")
     #   return

    #time = trans.variables['time'][:]  # Supposons qu'il y a une variable 'time' pour l'axe des x

    # Exemple de tracé
    #plt.figure()
    #plt.plot(time, transport_data, label='Transport Data')
    #plt.xlabel('Time')
    #plt.ylabel('Transport')
    #plt.title('Transport Data Over Time')
    #plt.legend()
    #plt.show()
    plotdiag.plot_streamfunctions(trans, name='simulated', basename='', obs=None,lw=4)
    trans.close()


main()


import xarray as xr

# Étape 1 : Lire le fichier NetCDF existant et extraire la variable 'vflx'
file_path = 'RapidMoc_3000-3000_natl_meridional_transports_at_26N.nc'
ds = xr.open_dataset(file_path)
vflx = ds['vflx']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_vflx = xr.Dataset({
        'vflx': vflx
        })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCD
v = 'vflx.nc'
new_vflx.to_netcdf(v)

print(f"Le nouveau fichier NetCDF contenant seulement 'vflx' a été créé : {v}")






saln = ds['salnlvl']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_s = xr.Dataset({
            'salnlvl': saln
                    })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCDF
s = 'saln.nc'
new_s.to_netcdf(s)

print(f"Le nouveau fichier NetCDF contenant seulement 'saln' a été créé : {s}")







temp = ds['templvl']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_t = xr.Dataset({
            'templvl': temp
                    })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCDF
t = 'temp.nc'
new_t.to_netcdf(t)

print(f"Le nouveau fichier NetCDF contenant seulement 'temp' a été créé : {t}")







dp = ds['dp']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_dp = xr.Dataset({
                'dp': dp
                        })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCDF
dp_f = 'dp.nc'
new_dp.to_netcdf(dp_f)

print(f"Le nouveau fichier NetCDF contenant seulement 'dp' a été créé : {dp_f}")








# Étape 1 : Lire le fichier NetCDF existant et extraire la variable 'vflx'
file_path2 = 'NBF1850OC_ORGANIC1.micom.hm.3000-01.nc'
dat = xr.open_dataset(file_path2)
taux = dat['ztx']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_taux = xr.Dataset({
            'ztx': taux
                    })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCD
tx = 'taux.nc'
new_taux.to_netcdf(tx)

print(f"Le nouveau fichier NetCDF contenant seulement 'taux' a été créé : {tx}")




d=xr.open_dataset('orga_hy/NBF1850OC_ORGANIC1.micom.hy.3000.nc')
vvellvl = d['vvellvl']

# Étape 2 : Créer un nouvel ensemble de données avec la variable 'vflx'
new_v = xr.Dataset({
                'vvellvl': vvellvl
                                    })

# Étape 3 : Enregistrer le nouvel ensemble de données dans un nouveau fichier NetCD
vv = 'vfile.nc'
new_v.to_netcdf(vv)

print(f"Le nouveau fichier NetCDF contenant seulement 'vvellvl' a été créé : {vv}")

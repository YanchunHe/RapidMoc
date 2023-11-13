#!/bin/bash
set -e

# This is tested on Fram with the following environment

#conda activate mypython

year1s=()
yearns=()

## historical
#year1s+=(1980 1990 2000 2010)
#yearns+=(1989 1999 2009 2014)

#mip=keyclim-v2
#expid=hist-eddy
#expid=hist-snow
#expid=hist-cloud2
#expid=hist-iceSheet3
#expid=hist-ozone
#version=v20230612
#expid=hist-piAerOxid
#version=v20210521

#KeyCLIM_CMOR=/projects/NS9252K/cmorout/NorESM2-MM/${expid}/${version}

#mip=cmip6-v2
#expid=historical
#version=v20191108
#KeyCLIM_CMOR=/projects/NS9034K/CMIP6/.cmorout/NorESM2-MM/${expid}/${version}

## Scenario
#year1s+=(2015)
#yearns+=(2020)
year1s+=(2071 2081 2091)
yearns+=(2080 2090 2100)
#year1s+=(2071 2081 2091)
#yearns+=(2080 2090 2099)

mip=keyclim-v2
expid=ssp585-eddy
#expid=ssp585-snow
#expid=ssp585-cloud2
#expid=ssp585-iceSheet3
#expid=ssp585-ozone
#expid=ssp585-piAerOxid
version=v20230612
KeyCLIM_CMOR=/projects/NS9252K/cmorout/NorESM2-MM/${expid}/${version}

#mip=cmip6-v2
#expid=ssp585
#version=v20230616
#KeyCLIM_CMOR=/projects/NS9034K/CMIP6/.cmorout/NorESM2-MM/${expid}/${version}

sed -i "65s+outdir=.*+outdir=outdir/$mip/${expid}+" ./etc/config.ini.keyclim
sed -i "64s+name=.*+name=$expid+" ./etc/config.ini.keyclim

datapath=$KeyCLIM_CMOR
outpath=~/projects/KeyCLIM/RapidMoc/outdir/${mip}/${expid}
[ ! -d $outpath ] && mkdir -p $outpath
#
mkdir -p $outpath
rm -f filelist
cp etc/config.ini.keyclim $outpath/

outputfiles=()
for (( i = 0; i < ${#year1s[*]}; i++ )); do
  year1=${year1s[i]};yearn=${yearns[i]}
  yeartag=${year1}01-${yearn}12
  outputfiles+=(${expid}_${year1}-${yearn}_natl_meridional_transports_at_26N.nc)
  #./run_rapidmoc.py ./etc/config.ini.keyclim \
    #$datapath/thetao_Omon_NorESM2-MM_${expid}_r1i1p1f1_gr_${yeartag}.nc \
    #$datapath/so_Omon_NorESM2-MM_${expid}_r1i1p1f1_gr_${yeartag}.nc \
    #$datapath/tauuo_Omon_NorESM2-MM_${expid}_r1i1p1f1_gn_${yeartag}.nc \
    #$datapath/vo_Omon_NorESM2-MM_${expid}_r1i1p1f1_gr_${yeartag}.nc
done

echo ${outputfiles[*]} |ncrcat -h -O -p $outpath --output=${outpath}/${expid}_${year1s[0]}-${yearns[-1]}_natl_meridional_transports_at_26N.nc

#cdo selyear,1985/2014 ${outpath}/${expid}_${year1s[0]}-${yearns[-1]}_natl_meridional_transports_at_26N.nc ${outpath}/${expid}_1985-2014_natl_meridional_transports_at_26N.nc

rm -f ${outputfiles[*]}

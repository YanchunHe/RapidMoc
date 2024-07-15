#!/bin/bash
set -e

# This is tested on Fram with the following environment

#SBATCH --account=nn1002k
#SBATCH --job-name=noresm2netcdf4
#SBATCH --qos=preproc
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

module purge
module load NCO/5.0.3-intel-2021b
#module load Python/3.9.6-GCCcore-11.2.0
#module load matplotlib/3.4.3-intel-2021b
#module load xarray/2022.6.0-foss-2022a

CASE=NBF1850OC_ORGANIC1

yyyy1=3000
yyyy2=3009

datapath=/scratch/yanchun/ORGANIC/NBF1850OC_ORGANIC1/ocn/hist
outpath=/projects/NS9015K/www/output/organic/NBF1850OC_ORGANIC1/rapidmoc2024
outpath2=/projects/NS9015K/www/output/rapidmoc/NBF1850OC_ORGANIC1/2024

mkdir -p $outpath $outpath2
rm -f filelist
cp etc/config.ini.organic $outpath/

echo years from $yyyy1 to $yyyy2
for (( yr = $yyyy1; yr <= $yyyy2; yr++ )); do
    echo $yr
    annual_file=RapidMoc_${yr}-${yr}_natl_meridional_transports_at_26N.nc

    if [ ! -f ${outpath}/${annual_file} ]; then
        sed -i "74s/date_format=.*/date_format=%Y/" ./etc/config.ini.organic
        ncks -h -O -v temp,saln,vflx,vvellvl ${datapath}/NBF1850OC_ORGANIC1.micom.hy.${yr}.nc ./hy.3d.nc
        ncks -h -O -v ztx ${datapath}/NBF1850OC_ORGANIC1.micom.hm.${yr}-01.nc ./hm.2d.nc
        ncks -h -A -v plat,plon,vlat,vlon ~/tools/data/tnx1v1/grid.nc4 ./hy.3d.nc
        ncks -h -A -v ulat,ulon ~/tools/data/tnx1v1/grid.nc4 ./hm.2d.nc

        ./run_rapidmoc.py ./etc/config.ini.organic ./hy.3d.nc ./hy.3d.nc ./hm.2d.nc ./hy.3d.nc ./hy.3d.nc ./dp.nc
        
        # use ekman transport calculated by monthly wind stress to overide ekman transport by yearly ztx
        sed -i "74s/date_format=.*/date_format=%Y%m/" ./etc/config.ini.organic
        for mon in `seq -f %02g 1 12`;do
            month_file=RapidMoc_${yr}${mon}-${yr}${mon}_natl_meridional_transports_at_26N.nc
            if [[ ! -f ${outpath}/${month_file} ]]; then
                ncks -h -O -v ztx ${datapath}/NBF1850OC_ORGANIC1.micom.hm.${yr}-${mon}.nc ./hm.2d.nc
                ncks -h -A -v ulat,ulon ~/tools/data/tnx1v1/grid.nc4 ./hm.2d.nc
                ncks -h -A -v time ./hm.2d.nc ./hy.3d.nc
                echo "run on monthly file $month_file "
                ./run_rapidmoc.py ./etc/config.ini.organic ./hy.3d.nc ./hy.3d.nc ./hm.2d.nc ./hy.3d.nc ./hy.3d.nc ./dp.nc
            fi
        done

        ncra -h -A -v ekman,sf_ek,q_ek,fw_ek ${outpath}/RapidMoc_${yr}*-${yr}*_natl_meridional_transports_at_26N.nc ${outpath}/${annual_file}
        ncap2 -O -s 'fw_sum_rapid=fw_fc+fw_ek+fw_mo' ${outpath}/${annual_file} ${outpath}/${annual_file}
        ncap2 -O -s 'q_sum_rapid=q_fc+q_ek+q_mo' ${outpath}/${annual_file} ${outpath}/${annual_file}
        #rm -f ${outpath}/RapidMoc_${yr}??-${yr}??_natl_meridional_transports_at_26N.nc
    fi
    
    echo ${outpath}/${annual_file} >>filelist
    rm -f hy.3d.nc hm.2d.nc
done

cat filelist |ncrcat -h -O --output=${outpath}/RapidMoc_${yyyy1}-${yyyy2}_natl_meridional_transports_at_26N.nc
for fname in $(cat filelist)
do
  mv $fname -t $outpath2/
done


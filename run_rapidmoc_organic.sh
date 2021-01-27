#!/bin/bash
#set -ex

# This is tested on Fram with the following environment

#SBATCH --account=nn1002k
#SBATCH --job-name=noresm2netcdf4
#SBATCH --qos=preproc
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

#module load NCO/4.6.6-intel-2017a
#module load Python/2.7.13-intel-2017a

CASE=NBF1850OC_ORGANIC1

yyyy1=1001
yyyy2=3150

datapath=/projects/NS9015K/yanchun/archive/NBF1850OC_ORGANIC1/ocn/hist
outpath=/projects/NS9015K/www/output/organic/NBF1850OC_ORGANIC1/rapidmoc
mkdir -p $outpath

rm -f filelist

echo years from $yyyy1 to $yyyy2
for (( yr = $yyyy1; yr <= $yyyy2; yr++ )); do
    echo $yr
    annual_file=RapidMoc_${yr}-${yr}_natl_meridional_transports_at_26N.nc

    if [ ! -f ${outpath}/${annual_file} ]; then
        sed -i "74s/date_format.*/date_format=%Y/" ./etc/config.ini.organic
        ncks -h -O -v templvl,salnlvl,vvellvl ${datapath}/NBF1850OC_ORGANIC1.micom.hy.${yr}.nc ./hy.3d.nc
        ncks -h -O -v ztx ${datapath}/NBF1850OC_ORGANIC1.micom.hm.${yr}-01.nc ./hm.2d.nc
        ncks -h -A -v plat,plon,vlat,vlon ~/tools/data/grid_tnx1v1.nc ./hy.3d.nc
        ncks -h -A -v ulat,ulon ~/tools/data/grid_tnx1v1.nc ./hm.2d.nc

        #ln -sf ${outpath}/hm.2d.nc .
        #ln -sf ${outpath}/hy.3d.nc .

        ./run_rapidmoc.py ./etc/config.ini.organic ./hy.3d.nc ./hy.3d.nc ./hm.2d.nc ./hy.3d.nc

        sed -i "74s/%Y/%Y%m/" ./etc/config.ini.organic
        for mon in `seq -f %02g 1 12`;do
            month_file=RapidMoc_${yr}${mon}-${yr}${mon}_natl_meridional_transports_at_26N.nc
            if [[ ! -f ${outpath}/${month_file} ]]; then
                ncks -h -O -v ztx ${datapath}/NBF1850OC_ORGANIC1.micom.hm.${yr}-${mon}.nc ./hm.2d.nc
                ncks -h -A -v ulat,ulon ~/tools/data/grid_tnx1v1.nc ./hm.2d.nc
                ncks -h -A -v time ./hm.2d.nc ./hy.3d.nc
                echo "run on monthly file $month_file "
                ./run_rapidmoc.py ./etc/config.ini.organic ./hy.3d.nc ./hy.3d.nc ./hm.2d.nc ./hy.3d.nc
            fi
        done

        ncra -h -A -v ekman,sf_ek,q_ek,fw_ek ${outpath}/RapidMoc_${yr}??-${yr}??_natl_meridional_transports_at_26N.nc ${outpath}/${annual_file}
        ncap2 -O -s 'fw_sum_rapid=fw_fc+fw_ek+fw_mo' ${outpath}/${annual_file} ${outpath}/${annual_file}
        ncap2 -O -s 'q_sum_rapid=q_fc+q_ek+q_mo' ${outpath}/${annual_file} ${outpath}/${annual_file}
        rm -f ${outpath}/RapidMoc_${yr}??-${yr}??_natl_meridional_transports_at_26N.nc
    fi
    
    echo ${outpath}/${annual_file} >>filelist
    rm -f hy.3d.nc hm.2d.nc
done

cat filelist |ncrcat -h -O --output=${outpath}/RapidMoc_${yyyy1}-${yyyy2}_natl_meridional_transports_at_26N.nc


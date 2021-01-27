#!/bin/bash
#set -ex

#SBATCH --account=nn1002k
#SBATCH --job-name=noresm2netcdf4
#SBATCH --qos=preproc
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1

module load NCO/4.6.6-intel-2017a
module load Python/2.7.13-intel-2017a

CASE=NOI20C_sr10m60d52

yyyy1=557
yyyy2=695

datapath=/tos-project2/NS9015K/yanchun/archive/${CASE}/ocn/hist
outpath=/cluster/projects/nn2343k/rapidmoc/${CASE}/

ln -sf ${outpath}/data.nc .
rm -f filelist

sed -i "74s/date_format.*/date_format=%Y%m/" ./etc/config.ini.20cr
echo years from $yyyy1 to $yyyy2
for (( yr = $yyyy1; yr <= $yyyy2; yr++ )); do
    yyyy=`printf "%04i\n" ${yr}`
    echo $yyyy
    for mm in `seq -f %02g 1 12`;do
        month_file=RapidMoc_${yr}${mm}-${yr}${mm}_natl_meridional_transports_at_26N.nc
        echo ${outpath}/${month_file} >>filelist
        if [[ ! -f ${outpath}/${month_file} ]]; then
            ncks -h -O -v templvl,salnlvl,vvellvl ${datapath}/${CASE}.micom.hm.${yyyy}-${mm}.nc ${outpath}/data.nc
            ncks -h -A -v ztx ${datapath}/${CASE}.micom.hm.${yyyy}-${mm}.nc ${outpath}/data.nc
            ncks -h -A -v plat,plon,ulat,ulon,vlat,vlon ~/tools/data/grid_tnx1v1.nc ${outpath}/data.nc
            ./run_rapidmoc.py ./etc/config.ini.20cr "${outpath}/data.nc" "${outpath}/data.nc" "${outpath}/data.nc" "${outpath}/data.nc"
        fi
    done
done
rm -f ${outpath}/data.nc
rm -f ${outpath}/*.tmp 

mkdir -p ${HOME}/output/20CR/${CASE}/rapidmoc
cat filelist |ncrcat -h -O --output=${HOME}/output/20CR/${CASE}/rapidmoc/RapidMoc_${yyyy1}-${yyyy2}_natl_meridional_transports_at_26N.nc

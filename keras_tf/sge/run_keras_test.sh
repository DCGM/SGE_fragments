#!/bin/bash
#$ -S /bin/bash
#$ -e /mnt/matylda1/ibrejcha/devel/SGE_fragments/keras_tf/logs
#$ -o /mnt/matylda1/ibrejcha/devel/SGE_fragments/keras_tf/logs
#$ -q long.q@@gpu
#$ -l matylda1=5,ram_free=6000M,mem_free=6000M,gpu=1
#$ -N tmnlp
##$ -pe smp 10

PROJDIR="/mnt/matylda1/ibrejcha/devel/SGE_fragments/keras_tf"

#export OMP_NUM_THREADS=$NSLOTS

if [ -z "$PROJDIR" ]; then
	echo "Project folder not set. Exitting."
	exit
fi

cd $PROJDIR || exit

source TENV/bin/activate
export LD_LIBRARY_PATH="/usr/local/share/cuda-8.0.61/lib64:$LD_LIBRARY_PATH"

python mnist_mlp.py

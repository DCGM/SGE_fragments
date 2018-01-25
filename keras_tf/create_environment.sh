#!/bin/bash
#Author Jan Brejcha 2018

virtualenv TENV 						#create virtual environment named TENV
source TENV/bin/activate 				#acrtivate the virtual environment

#now we are in the virtual environment
pip install --upgrade tensorflow-gpu 	#install tensorflow with GPU support

pip install keras opencv-python			#install other dependencies

#create logs directory for SGE
mkdir -p logs

#download mnist data
LD_LIBRARY_PATH="/usr/local/share/cuda-8.0.61/lib64:$LD_LIBRARY_PATH" python prepareData.py

#deactivate the virtual environment
deactivate



SGE Keras Tensorflow Example
============================

This example shows how to install Tensorflow&Keras inside a Python's virtual environment and how to run simple Keras code on Merlin SGE cluster. 

1. To create a virtual environment with Tensorflow and Keras, run `./create_environment.sh`
2. Edit `sge/run_keras_test.sh` and set full path to the logs directory (lines 3, 4) and `PROJDIR` (line 5).
3. Run the code on SGE `qsub sge/run_keras_test.sh`
4. See output in the log files.


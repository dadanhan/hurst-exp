# hurst-exp
Estimating Hurst exponents varying in time using neural networks for time series

Ensure you have Python3, tensorflow 1.14 and keras installed.
FBM package used is from https://github.com/crflynn/fbm.

model3densediff_n13.h5 is the saved tensorflow model that has been pre-trained
singleexpest.py is the python script to use to classify a single trajectory with a single Hurst exponent.
multiexpest.py is the python script to use to classify a single trajectory with varying Hurst exponent by using a symmetric moving window.

'''
Ryan Pierce
ID: 2317826
EECS 690 - Introduction to Machine Learning, Python Project 1
Run program to check that libs are installed
Also prints their current versions
Program uses f-strings, so be sure to use python3 in command-line
'''

# Python Version
import sys
print (f'Python {sys.version}')

# scipy
import scipy
print (f'scipy: {scipy.__version__}')

# numpy
import numpy
print (f'numpy: {numpy.__version__}')

# matplotlib
import matplotlib
print (f'matplotlib: {matplotlib.__version__}')

# pandas
import pandas
print (f'pandas: {pandas.__version__}')

# scikit-learn
import sklearn
print (f'scikit-learn: {sklearn.__version__}')

print ('\n')
print ('Hello World!')
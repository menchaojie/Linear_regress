import os
import numpy as np

f = os.path.join('.','linaer_regression.npy')

(X, Y) = np.load(f)

print(X)


import numpy as np


Limits = np.array([87.2108, 93.5393, 36.6461, 9.8927, 8.9563, 5.07])
curr = np.array([4.7744317 , 6.74911779, 2.8461758 , 1.31314992 ,0.    ,     0.42223216])

violation_percentage = curr / Limits

print(violation_percentage * 100)
# 11.5% backoff
print(Limits * 0.93)
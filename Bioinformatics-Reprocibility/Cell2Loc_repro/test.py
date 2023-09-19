import numpy as np
from scipy.stats import pearsonr


x = np.array([1, 2, 4, 3, 0, 0, 0], dtype=np.float32)
y = np.array([1, 2, 3, 4, 0, 0, 0], dtype=np.float32)
y_s = np.array([2, 4, 6, 8, 0, 0, 0], dtype=np.float32)
y_t = np.array([6, 7, 8, 9, 0, 0, 0], dtype=np.float32)


print(pearsonr(x, y))
print(pearsonr(x, y_s))
print(pearsonr(x, y_t))


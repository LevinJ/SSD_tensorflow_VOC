import numpy as np
from sklearn.metrics import average_precision_score
 
y_true = np.array([[0, 0, 1, 1],[0, 0, 1, 1]]).T
y_scores = np.array([[0.1, 0.4, 0.35, 0.6],[0.1, 0.1, 0.35, 0.3]]).T
 
# y_true = np.array([0, 0, 1, 1])
# y_scores = np.array([0.1, 0.1, 0.35, 0.8])
 
 
res = average_precision_score(y_true, y_scores) 
print(res)










import numpy as np
a=np.array([[[255, 255] ,[255, 255]],[[0,0],[0,0]]])
a=a.astype(np.float64) / 255.0
print(a)
H, W, D = a.shape
H = (H // 8) * 8
W = (W // 8) * 8
print(H)
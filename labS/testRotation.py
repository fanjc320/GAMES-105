from scipy.spatial.transform import Rotation as R
import numpy as np

vector = np.array([1, 0, 0])
r = R.from_rotvec([0, 0, np.pi/2])
# r.as_matrix() # // 有没有都一样

res = r.apply(vector)
print("res:" , res) # [0, 1, 0]

vectorz = np.array([0, 0, 1])
ryz = R.from_rotvec([0, np.pi/2, np.pi/2])
resz = ryz.apply(vectorz)
print("resz:" , resz, " shape:", resz.shape) # resz: [0.56264006 0.80284993 0.19715007]  shape: (3,)
# resz02 = resz[0, 2] IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
resz02 = resz[[0, 2]]
print("resz02:" , resz02, " shape:", resz02.shape)
# resz02 = resz.flatten()[0,2] # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
resz02 = resz.flatten()[[0,2]] # resz02: [0.56264006 0.19715007]  shape: (2,)
print("resz02:" , resz02, " shape:", resz02.shape)
# 上面两个输出结果相同

# rxyz = R.from_euler('xyz', [
# [0, 0, 90],
# [45, 30, 60]], degrees=True)

rxyz = R.from_euler('xyz', [

[0, 0, 90]], degrees=True)
vectors = [
[1, 0, 0],
[0, 1, 0],
[0, 0, 1]]

res_xyz = rxyz.apply(vectors)
print("res_xyz:" , res_xyz)


rxyz = R.from_euler('xyz', [
[0, 0, 90]], degrees=True)

vectors = [
[1, 1, 0],
[0, 1, 1],
[1, 0, 1]
]

res_xyz = rxyz.apply(vectors)
print("res_xyz:" , res_xyz)

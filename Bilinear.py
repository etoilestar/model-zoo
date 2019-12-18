import numpy as np

def bilinear(array, dsize=(30,20)):
    array_new = np.ones(dsize)
    x, y = np.where(array_new)
    x = x.reshape(dsize)
    y = y.reshape(dsize)
    init_shape = array.shape
    ratio_x = (init_shape[0]-1) / (dsize[0]-1)
    ratio_y = (init_shape[1]-1) / (dsize[1]-1)
    qx, qy = ratio_x*x, ratio_y*y
    x_floor, y_floor = np.floor(qx), np.floor(qy)
    x_ceil, y_ceil = np.ceil(qx), np.ceil(qy)
    p1 = array[x_floor.astype(int), y_floor.astype(int)]
    p2 = array[x_floor.astype(int), y_ceil.astype(int)]
    p3 = array[x_ceil.astype(int), y_floor.astype(int)]
    p4 = array[x_ceil.astype(int), y_ceil.astype(int)]
    array_new = ((qx-x_floor)*p3 + (x_ceil-qx)*p1)*(y_ceil-qy)+((qx-x_floor)*p4 + (x_ceil-qx)*p2)*(qy-y_floor)
    array_new = np.where(x_ceil==x_floor, p1*(y_ceil-qy) + p2*(qy-y_floor), array_new)
    array_new = np.where(y_ceil==y_floor, p1*(x_ceil-qx) + p3*(qx-x_floor), array_new)
    array_new = np.where(np.all(np.concatenate((np.expand_dims(y_ceil==y_floor, -1),np.expand_dims(x_ceil==x_floor,-1)), -1),-1), p1, array_new)
    return array_new

if __name__ == '__main__':
    array = np.random.rand(16).reshape(4,4)
    print(bilinear(array, dsize=(16,16)))
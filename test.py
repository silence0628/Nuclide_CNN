# coding:utf-8
"""
    众多程序测试
"""
import scipy.io as scio
import os

dir = './save_features/'

def visualShape(feature_dir):
    Lists = os.listdir(dir)
    mat = scio.loadmat(dir + Lists[-1])
    print('{:*<70}'.format(str(mat['conv1'].shape)))
    print('{:*<70}'.format(str(mat['pool1'].shape)))
    print('{:*<70}'.format(str(mat['conv2'].shape)))
    print('{:*<70}'.format(str(mat['pool2'].shape)))
    print('{:*<70}'.format(str(mat['local3'].shape)))
    print('{:*<70}'.format(str(mat['local4'].shape)))
    print('{:*<70}'.format(str(mat['softmax'].shape)))

visualShape(dir)
# 2D
# """
#     (20, 32, 32, 16)
#     (20, 16, 16, 16)
#     (20, 16, 16, 16)
#     (20, 16, 16, 16)
#     (20, 128)
#     (20, 128)
#     (20, 6)
# """

# 1D
# """
# (20, 1, 1024, 16)
# (20, 1, 512, 16)
# (20, 1, 512, 16)
# (20, 1, 512, 16)
# (20, 128)
# (20, 128)
# (20, 6)
# """
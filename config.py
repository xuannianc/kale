import os
import os.path as osp
DATASET_DIR = '/home/adam/.keras/datasets/synthetic_chinese_string'
TRAIN_DIR = osp.join(DATASET_DIR, 'train')
TEST_DIR = osp.join(DATASET_DIR, 'test')
# train.txt 一共 3279606 行
# test.txt 一共 364400 行
# images 目录在移除掉 test.txt 里面的文件后, 重命名为 train
IMAGE_DIR = osp.join(DATASET_DIR, 'images')
TRAIN_TXT_PATH = osp.join(DATASET_DIR, 'train.txt')
TEST_TXT_PATH = osp.join(DATASET_DIR, 'test.txt')
TEST_DB_PATH = osp.join(DATASET_DIR, 'synthetic_test_0829_364400.hdf5')
TRAIN_DB_PATH = osp.join(DATASET_DIR, 'synthetic_train_0829_3000000.hdf5')
VALIDATION_DB_PATH = osp.join(DATASET_DIR, 'synthetic_validation_0829_279600.hdf5')

import pandas as pd
import numpy as np
from utils.process_labels import encode_labels
from pyecharts import Line, Pie
from pyecharts_snapshot.main import make_a_snapshot
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main(label_paths):
    out_array = AverageMeter()
    area_dict = {}
    for i in range(8):
        area_dict[i] = 0
    # num_dict = {}
    for s, path in enumerate(tqdm(label_paths)):
        ori_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        encode_mask = encode_labels(ori_mask).astype('uint8')
        get_one_hot = np.eye(8)[encode_mask]
        get_one_hot = np.sum(np.sum(get_one_hot, 0), 0)
        out_array.update(get_one_hot.copy())
    return out_array.avg/100

if __name__ == '__main__':
    data_dir = './data_list/train.csv'
    total_list = pd.read_csv(data_dir)
    label_paths = total_list['label']
    thread_num = 32
    n = int(math.ceil(len(label_paths) / float(thread_num)))
    pool = multiprocessing.Pool(processes=thread_num)
    result = []
    for i in tqdm(range(0, len(label_paths), n)):
        result.append(pool.apply_async(main, (label_paths[i: i+n],)))
    for step, r in enumerate(tqdm(result)):
        arr = r.get()
        if step == 0:
            out_array = arr
        else:
            out_array += arr
    out_array /= thread_num
    explode=[]
    l = []

    for i in range(8):
        explode.append(0.01)
        l.append(i)

    values = list(out_array)
    pie = Pie('sample distribution')
    pie.add('class', l, values)
    pie.render('test.html')
    make_a_snapshot('test.html', 'test.pdf')
    # plt.rcParams['font.sans-serif']='SimHei'
    # plt.figure(figsize=(15,15))
    # print(values)
    # plt.pie(values,explode=explode,labels=l,autopct='%1.1f%%')
    
    # plt.title('统计面积')#绘制标题
    # plt.savefig('./statistic.png')#保存图片
    # plt.rcParams['font.sans-serif']='SimHei'
    # plt.figure(figsize=(15,15))
    # plt.pie(values[1:],explode=explode[1:],labels=l[1:],autopct='%1.1f%%')
    
    # plt.title('统计面积')#绘制标题
    # plt.savefig('./statistic_no_zero.png')#保存图片
    pool.close()
    pool.join()

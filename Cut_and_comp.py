from __future__ import (
    division,
    print_function,
)

import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.siamese import Siamese as siamese
from utils.utils import letterbox_image, preprocess_input, cvtColor, show_config


import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os

from siamese import Siamese

def getFileList(dir,Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist

def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = Image.open(path)

    print("This image's size: {}".format(img.size))   #  (W, H)
    
    plt.figure("Image Contrast")

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)
    plt.axis('off')

    box = (left, upper, right, lower)
    roi = img.crop(box)

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(roi)
    plt.axis('off')
    plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = Image.open(path)  # 打开图像
    box = (left, upper, right, lower)
    roi = img.crop(box)

    # 保存截取的图片
    roi.save(save_path)
    

def txt2array(txt_path, delimiter):
    #---
    # 功能：读取只包含数字的txt文件，并转化为array形式
    # txt_path：txt的路径；delimiter：数据之间的分隔符
    #---
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(delimiter)
        temp = list(map(float, data_split))
        data_list.append(temp)

    return data_list



def main():

    model = Siamese()
    
    # loading astronaut image
    # img = skimage.data.astronaut()
    # ----demo1：切片一次循环的工作，假设循环进行到0001.jpg---- #
    # ----测试完成后扩写为遍历文件夹中所有图片---- #
    
    #先把标记框都读入
    annotation_file = "img/BlurCar2/groundtruth_rect.txt"
    annotation = txt2array(annotation_file,',')
    count = 0
    template_require = True
    mask_list = []
    for image_path in getFileList("img/BlurCar2/img",[],"jpg"):
        #image_path = 'img/BlurCar2/img/0001.jpg'  # 随意选择一张图片进行测试
        print('===== This is frame: ',count,' =====')
        image_root,image_name = os.path.split(image_path)
        img = Image.open(image_path)
        img = np.array(img)
        
        
        #以下程序为依次读入所有注释
        #i = 1
        # for rect in annotation:
        #     if i<10:
        #         img_path = os.path.join(image_root,'000'+str(i)+'.jpg')
        #     elif i<=99:
        #         img_path = os.path.join(image_root,'00'+str(i)+'.jpg')
        #     elif i<=999:
        #         img_path = os.path.join(image_root,'0'+str(i)+'.jpg')
        #     save_path = os.path.join(image_root+'/target/','body_target'+str(i)+'.jpg')
        #     image_cut_save(img_path,int(rect[0]),int(rect[1]),int(rect[0])+int(rect[2]),int(rect[1])+int(rect[3]),save_path)
        #     i = i+1
        
        #获取每一张图片的target_box
        target_box = (annotation[count][0],annotation[count][1],
                    annotation[count][0]+annotation[count][2],annotation[count][1]+annotation[count][3])
        
        # perform selective search
        # img_lbl, regions = selectivesearch.selective_search(
        #     img, scale=500, sigma=0.9, min_size=10)

        # candidates = set()
        # for r in regions:
        #     # excluding same rectangle (with different segments)
        #     if r['rect'] in candidates:
        #         continue
        #     # excluding regions smaller than 2000 pixels
        #     if r['size'] < 2000:
        #         continue
        #     # distorted rects
        #     x, y, w, h = r['rect']
        #     if w / h > 1.2 or h / w > 1.2:
        #         continue
        #     candidates.add(r['rect'])
        #以上部分完成了SS算法，其中，分割框都存在了candidates这一列表中

        #可视化部分，暂时不用
        # # draw rectangles on the original image
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(img)
        
        #对于要处理的每帧图片需要模板比较相似度，相似度需要在第一帧获得。
        image = Image.open(image_path)
        if template_require:
            target_box_enlarged = (target_box[0],target_box[1],target_box[2],target_box[3]) #需不需要放大一圈？还无定论
            template = target = image.crop(target_box_enlarged)
            template_require = False
        
        #先获取target，计算遮蔽度
        target = image.crop(target_box)
        mask = 0
        mask = model.detect_image(template,target)
        mask_list.append(1 - mask.item())
        print('mask: ',1 - mask.item())
        num = 1
        # for x, y, w, h in candidates:
        #     #以下是SS切割图像部分
        #     #分别定位图片目标框和候选框
        #     block_box = (x, y, x+w, y+h)  
        #     block = image.crop(block_box)
            
        #     block_point = [(block_box[0]+block_box[2])/2,(block_box[1]+block_box[3])/2]
        #     target_point = [(target_box[0]+target_box[2])/2,(target_box[1]+target_box[3])/2]
        #     distance = math.sqrt((block_point[0]-target_point[0])**2 +(block_point[1]-target_point[1])**2)
        #     in_box = False
        #     if target_box[0] < block_point[0] < target_box[2] and target_box[1] < block_point[1] < target_box[3]:
        #         in_box = True
                
        #     probability = model.detect_image(block,target)
        #     mask = model.detect_image(template,target)
        #     print('--------')
        #     print('block: ',x, y, w, h)
        #     print('similarity: ',probability.item())
        #     print('distance by pixel: ',distance)
        #     print('overlap with target: ',in_box)
        #     print('--------')
        #     #image_cut_save(image_path, x, y, x+w, y+h, image_name)
        #     num = num+1
            
        #     #以下使用孪生网络比较切下来的块和similarity
            
            
        #     #以下暂时用不着（画红框）
        #     # rect = mpatches.Rectangle(
        #     #     (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        #     # ax.add_patch(rect)

        #plt.show()
        # if count >= 10:
        #     break
        count = count + 1
    
    F = open(r'mask.txt','w')
    for i in mask_list:
        F.write(str(i)+'\n')
    F.close()
if __name__ == '__main__':
    main()

import cv2
import numpy as np
import os
import sys
import shutil

from tqdm import tqdm

def extract_frames(vid, dst):
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (144, 256)) # original size 720x1280
        img_path = os.path.join(dst, '%04d.png' % count)
        tqdm.write(img_path)
        cv2.imwrite(img_path, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    vidcap.release()  # 释放资源
    return count

def master(v_list, mode):

    #pack_path =  params.raw_videos_train  
    #pack_path =  params.raw_videos_val
    
    # 读取文件内容并获取行数
    with open(v_list, 'r') as videos:
        video_list = videos.readlines()
    
    pack_path =  '/public/home/shiyj2-group/video_localization/GAMa_dataset/videos/train/'
    base_path = '/public/home/shiyj2-group/video_localization/GAMa_dataset/BDD_frames/train/'
        
    if mode == 'val':
        pack_path =  '/public/home/shiyj2-group/video_localization/GAMa_dataset/videos/val/'
        base_path = '/public/home/shiyj2-group/video_localization/GAMa_dataset/BDD_frames/val/'

    cnt = 0
    total_videos = len(video_list)
    for sample in tqdm(video_list, desc="Extracting frames", leave=True, total=total_videos):
        sample = sample.rstrip('\n')
        
        v_name, num_frames = sample.split()
        cnt += 1
    
        sample_path = os.path.join(pack_path, v_name+'.mov')

        dst = os.path.join(base_path, v_name+'.mov')

        # 检查文件夹是否存在，如果存在则删除
        if os.path.exists(dst):
            # 获取文件夹中的所有文件
            # files = os.listdir(dst)
            # num_files = len(files)
            # if num_files < 1100:
            #     shutil.rmtree(dst)
            #     # 创建新的文件夹
            #     os.makedirs(dst)
            #     extract_frames(sample_path, dst)
            # else:
            #     tqdm.write('exist: {0}'.format(dst))
            tqdm.write('exist: {0}'.format(dst))
        else:
            # 创建新的文件夹
            os.makedirs(dst)
            extract_frames(sample_path, dst)


    
    print(cnt)
    videos.close()

if __name__ == '__main__':
    mode = 'train'
    v_list = './list/' + mode + '_day.list'

    master(v_list, mode)


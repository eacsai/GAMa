import cv2
import numpy as np
import os
import sys
import shutil

def extract_frames(vid, dst):
    print(vid)
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    print(image.shape)
    print(success)
    count = 0
    while success:
        image = cv2.resize(image, (144, 256)) # original size 720x1280
        img_path = os.path.join(dst, '%04d.png' % count)
        print(img_path)
        cv2.imwrite(img_path, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    vidcap.release()  # 释放资源
    return count

def master(v_list, mode):

    #pack_path =  params.raw_videos_train  
    #pack_path =  params.raw_videos_val
    
    videos = open(v_list, 'r')
    
    pack_path =  '/public/home/v-wangqw/dataset/GAMa_dataset/videos/train/'
    base_path = '/public/home/v-wangqw/dataset/GAMa_dataset/BDD_frames/train/'
        
    if mode == 'val':
        pack_path =  '/public/home/v-wangqw/dataset/GAMa_dataset/videos/train/'
        base_path = '/public/home/v-wangqw/dataset/GAMa_dataset/BDD_frames/val/'

    cnt = 0
    for sample in videos:
        if cnt > 1000:
            break
        sample = sample.rstrip('\n')
        
        v_name, num_frames = sample.split()
          
        print(num_frames)
        print(v_name)
        cnt += 1
    
        sample_path = os.path.join(pack_path, v_name+'.mov')

        dst = os.path.join(base_path, v_name+'.mov')

        # 检查文件夹是否存在，如果存在则删除
        if os.path.exists(dst):
            shutil.rmtree(dst)  # 删除文件夹及其所有内容

        # 创建新的文件夹
        os.makedirs(dst)
        
        num_frames = extract_frames(sample_path, dst)


    
    print(cnt)
    videos.close()

if __name__ == '__main__':
    v_list = None
    
    mode = 'train'
    # if len(sys.argv) > 1:
    #     mode = str(sys.argv[1])
    # else:
    #     print('mode missing!')
    #     exit(0)

    v_list = './list/' + mode + '_day.list'

    master(v_list, mode)


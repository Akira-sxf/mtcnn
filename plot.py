import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse, Circle
import numpy as np
import os
 # 用于读取 detection，landmark 和 annotation
 # 返回字典列表，列表中每个元素为一个字典，包含的键有：图像名称，该图中脸的个数，脸的坐标
def getRes_det(fdet, fland):
    detFile = open(fdet,'r')
    det = detFile.readlines()
    detFile.close()
    if len(fland) > 0:
        landFile = open(fland, 'r')
        land = landFile.readlines()
        landFile.close()
    res = []
    for idx in range(len(det)):
        if '/' in det[idx]:
            im_res = {}
            im_res['name'] = det[idx].strip()
            num = int(det[idx+1])
            im_res['num'] = num
            coord = []
            landmark = []
            for i in range(idx+2,idx+2+num):
                coord_str = det[i].split()
                coord_float = [float(j) for j in coord_str]
                coord.append(coord_float)
                if len(fland) > 0:
                    landmark_str = land[i].split()
                    landmark_float = [float(k) for k in landmark_str]
                    landmark.append(landmark_float)
            im_res['coord'] = np.array(coord)
            im_res['landmark'] = np.array(landmark)
            res.append(im_res)
    return res
 # 在 res 中的每个元素中增加时间 键
def getRes_time(res, ftime):
    timeFile = open(ftime,'r')
    tm = timeFile.readlines()
    timeFile.close()
    for i in range(len(res)):
        res[i]['time'] = float(tm[i])
    return res
 # 画出 detection 和 annotation
def showRes(res, ann, thresh=0.5, class_name='face'):
    dets = res['coord']
    landmarks = res['landmark']
    inds = np.where(dets[:,-1]>=thresh)[0]
    im_name = os.path.join('data/FDDB','originalPics',res['name']+'.jpg')
    im = mpimg.imread(im_name)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(im, aspect='equal')
    x = []
    y = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        [x.append(landmarks[i][k]) for k in [0, 2, 4, 6, 8]]
        [y.append(landmarks[i][j]) for j in [1, 3, 5, 7, 9]]
#        ax.add_patch(Circle((10, 10), 10, edgecolor = 'r'))
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor='blue', linewidth=2)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')                
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),
                  fontsize=14)
    dets_ann = ann['coord']
    for i in range(len(dets_ann)):
        ellipse_ann = dets_ann[i, :5] #(r_vertical, r_horizontal, theta, c_x, c_y)
        ellipse = Ellipse(xy=(ellipse_ann[3], ellipse_ann[4]), 
                          width=ellipse_ann[0]*2, 
                          height=ellipse_ann[1]*2, 
                          angle=np.degrees(ellipse_ann[2]), 
                          edgecolor='g', fc='None', lw=2)
        ax.add_patch(ellipse)
    plt.scatter(x, y, s=50, color = '', edgecolor = 'r', linewidth=1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # plt.show()
    fig.savefig('data/FDDB_OUTPUT/image/'+res['name'].split('/')[-1]+'.png', dpi=300, transparent=True, pad_inches=0, bbox_inches='tight')
    
det_name = [] # detection file name
time_name = [] # detection time
ann_name = [] # annotation file name
land_name = []
for i in range(10):
    det_name.append('data/FDDB_OUTPUT/FDDB-det-fold-{:0>2}.txt'.format(i+1))
    time_name.append('FDDB-time-fold-{:0>2}.txt'.format(i+1))
    land_name.append('data/FDDB_OUTPUT/FDDB-landmark-fold-{:0>2}.txt'.format(i+1))
    ann_name.append('data/FDDB/FDDB-folds/FDDB-fold-{:0>2}-ellipseList.txt'.format(i+1))
res = []
ann = []
for i in range(10):
    res_det = getRes_det(det_name[i], land_name[i])
    #res_time = getRes_time(res_det, time_name[i])
    res.extend(res_det)
    ann_det = getRes_det(ann_name[i], '')
    ann.extend(ann_det)
 # 显示第 1000 幅图的效果
for i in range(20):
    # number = np.random.randint(len(res))
    number = i
    temp = res[number]
    temp_ann = ann[number]
    showRes(temp, temp_ann)
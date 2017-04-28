import numpy as np

mask = np.array([True, True]).reshape(-1,1)
mask = np.repeat(mask, [1,2], axis = 1)

res = np.where(mask,
     [[1, 2], [3, 4]],
     [[9, 8], [7, 6]])

print(res)

# a = np.array([1.0, 2.0, 3.0])
# b = 2.0

# print(a * b)

# x = np.arange(4)
# xx = x.reshape(4,1)
# 
# y = np.ones(5)

# x = np.array([1,2]).reshape((2,1))
# y =np.arange(4).reshape((1,4))
# 
# print(x-y)

# from numpy import array, argmin, sqrt, sum
# 
# observation = array([111.0,188.0])
# 
# codes = array([[102.0, 203.0],
#                [132.0, 193.0],
#                [45.0, 155.0],
#                [57.0, 173.0]])
# 
# # observation = observation.reshape((1,-1))
# # distance = np.sqrt((observation[:,0] - codes[:,0]) ** 2 + (observation[:,1] - codes[:,1]) ** 2)
# 
# diff = codes - observation
# distance = (diff **2).sum(axis=-1) 
# 
# min_ind = np.argmin(np.sqrt(distance))
# print(codes[min_ind])




def compute_jaccard(gt_bboxes, anchors):
    inter_ymin = np.maximum(gt_bboxes[:,:,0], anchors[:,:,0])
    inter_xmin = np.maximum(gt_bboxes[:,:,1], anchors[:,:,1])
    inter_ymax = np.minimum(gt_bboxes[:,:,2], anchors[:,:,2])
    inter_xmax = np.minimum(gt_bboxes[:,:,3], anchors[:,:,3])
    
    h = np.maximum(inter_ymax - inter_ymin, 0.)
    w = np.maximum(inter_xmax - inter_xmin, 0.)
    
    inter_area = h * w
    anchors_area = (anchors[:,:,3] - anchors[:,:,1]) * (anchors[:,:,2] - anchors[:,:,0])
    gt_bboxes_area = (gt_bboxes[:,:,3] - gt_bboxes[:,:,1]) * (gt_bboxes[:,:,2] - gt_bboxes[:,:,0])
    union_area = anchors_area - inter_area + gt_bboxes_area
    jaccard = inter_area/union_area
    print(jaccard)
    return jaccard

def match_achors(gt_labels, gt_bboxes, anchors,jaccard, matching_threshold = 0.5):
    num_bbox = gt_bboxes.shape[0]
    gt_anchor_labels = np.zeros(num_bbox)
    gt_anchor_ymins = np.zeros(num_bbox)
    gt_anchor_xmins = np.zeros(num_bbox)
    gt_anchor_ymaxs = np.ones(num_bbox)
    gt_anchor_xmaxs = np.ones(num_bbox)
    gt_anchor_bboxes = np.hstack([gt_anchor_ymins,gt_anchor_xmins,gt_anchor_ymaxs,gt_anchor_xmaxs])
    
    
    mask = np.max(jaccard, axis = 0) > matching_threshold
    
    gt_anchor_labels = np.where(mask, gt_labels, gt_anchor_labels)
    gt_anchor_bboxes = np.where(gt_anchor_bboxes, )
    
    return

gt_bboxes = np.array([[0,0,1,2],[1,0,3,4]]).reshape((-1,1,4))
gt_labels = np.array([1,2])
anchors = np.array([[100,100,105,105],[2,1,3,3.5],[0,0,10,10]]).reshape((1,-1,4))


jaccard = compute_jaccard(gt_bboxes, anchors)









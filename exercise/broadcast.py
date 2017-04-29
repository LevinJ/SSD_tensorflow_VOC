import numpy as np

inds = [2,5,8]

gt_anchors_labels= np.zeros(20)
gt_labels = np.arange(10)

maskids = [1,2]

gt_labels[maskids] = [110,100]

# b = gt_labels>5
# gt_labels[b] = [0,0,-1,-1]


gt_anchors_labels[inds] = gt_labels[inds]

print(gt_anchors_labels)
# mask = np.array([False, True]).reshape(-1,1)
# mask = np.repeat(mask, 2, axis = 1)
# 
# res = np.where(mask,
#      [[1, 2], [3, 4]],
#      [[9, 8], [7, 6]])
# 
# print(res)

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


prior_scaling=[0.1, 0.1, 0.2, 0.2] 

def compute_jaccard(gt_bboxes, anchors):
    
    gt_bboxes = np.reshape(gt_bboxes, (-1,1,4))
    anchors = np.reshape(anchors, (1,-1,4))
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
    num_anchors= jaccard.shape[1]

    gt_anchor_labels = np.zeros(num_anchors)
    gt_anchor_scores = np.zeros(num_anchors)
    gt_anchor_ymins = np.zeros(num_anchors)
    gt_anchor_xmins = np.zeros(num_anchors)
    gt_anchor_ymaxs = np.ones(num_anchors)
    gt_anchor_xmaxs = np.ones(num_anchors)
    gt_anchor_bboxes = np.stack([gt_anchor_ymins,gt_anchor_xmins,gt_anchor_ymaxs,gt_anchor_xmaxs], axis=-1)
    
    
    #match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).
    mask = np.max(jaccard, axis = 0) > matching_threshold
    mask_inds = np.argmax(jaccard, axis = 0)
    mask_inds = mask_inds[mask]
    gt_anchor_labels[mask] = gt_labels[mask_inds]
    gt_anchor_bboxes[mask] = gt_bboxes[mask_inds]
    gt_anchor_scores = np.max(jaccard, axis= 0)
    
    
    #matching each ground truth box to the default box with the best jaccard overlap
    inds = np.argmax(jaccard, axis = 1)
    
    gt_anchor_labels[inds] = gt_labels
    gt_anchor_bboxes[inds] = gt_bboxes
    gt_anchor_scores[inds] = jaccard[np.arange(jaccard.shape[0]),inds]
    
    # Transform to center / size.
    feat_cx = (gt_anchor_bboxes[:,3] + gt_anchor_bboxes[:,1]) / 2.
    feat_cy = (gt_anchor_bboxes[:,2] + gt_anchor_bboxes[:,0]) / 2.
    feat_w = gt_anchor_bboxes[:,3] - gt_anchor_bboxes[:,1]
    feat_h = gt_anchor_bboxes[:,2] - gt_anchor_bboxes[:,0]
    
    xref = (anchors[:,3] + anchors[:,1]) / 2.
    yref = (anchors[:,2] + anchors[:,0]) / 2.
    wref = anchors[:,3] - anchors[:,1]
    href = anchors[:,2] - anchors[:,0]
    
    
    # Encode features, convert ground truth bboxes to  shape offset relative to default boxes 
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_w = np.log(feat_w / wref) / prior_scaling[3]
    feat_h = np.log(feat_h / href) / prior_scaling[2]
    
    
    gt_anchor_bboxes = np.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    
    
    
    
    return gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores

gt_bboxes = np.array([[0,0,1,2],[1,0,3,4],[100,100,105,102.5]])
gt_labels = np.array([1,2,6])
anchors = np.array([[100,100,105,105],[2,1,3,3.5],[0,0,10,10],[0.5,0.5,0.8,1.5]])


jaccard = compute_jaccard(gt_bboxes, anchors)
gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores = match_achors(gt_labels, gt_bboxes, anchors,jaccard,matching_threshold = 0.01)
print(gt_anchor_bboxes)









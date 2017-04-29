import tensorflow as tf






prior_scaling=[0.1, 0.1, 0.2, 0.2] 

def compute_jaccard(gt_bboxes, anchors):
    
    gt_bboxes = tf.reshape(gt_bboxes, (-1,1,4))
    anchors = tf.reshape(anchors, (1,-1,4))
    
    inter_ymin = tf.maximum(gt_bboxes[:,:,0], anchors[:,:,0])
    inter_xmin = tf.maximum(gt_bboxes[:,:,1], anchors[:,:,1])
    inter_ymax = tf.minimum(gt_bboxes[:,:,2], anchors[:,:,2])
    inter_xmax = tf.minimum(gt_bboxes[:,:,3], anchors[:,:,3])
    
    h = tf.maximum(inter_ymax - inter_ymin, 0.)
    w = tf.maximum(inter_xmax - inter_xmin, 0.)
    
    inter_area = h * w
    anchors_area = (anchors[:,:,3] - anchors[:,:,1]) * (anchors[:,:,2] - anchors[:,:,0])
    gt_bboxes_area = (gt_bboxes[:,:,3] - gt_bboxes[:,:,1]) * (gt_bboxes[:,:,2] - gt_bboxes[:,:,0])
    union_area = anchors_area - inter_area + gt_bboxes_area
    jaccard = inter_area/union_area
    
    return jaccard

def match_achors(gt_labels, gt_bboxes, anchors,jaccard, matching_threshold = 0.5):
    num_anchors= jaccard.shape[1]

    gt_anchor_labels = tf.zeros(num_anchors, dtype=tf.int32)
    gt_anchor_scores = tf.zeros(num_anchors)
    gt_anchor_ymins = tf.zeros(num_anchors)
    gt_anchor_xmins = tf.zeros(num_anchors)
    gt_anchor_ymaxs = tf.ones(num_anchors)
    gt_anchor_xmaxs = tf.ones(num_anchors)
    gt_anchor_bboxes = tf.stack([gt_anchor_ymins,gt_anchor_xmins,gt_anchor_ymaxs,gt_anchor_xmaxs], axis=-1)
    
    
    #match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5).
    mask = tf.reduce_max (jaccard, axis = 0) > matching_threshold
    mask_inds = tf.argmax(jaccard, axis = 0)
    matched_labels = tf.gather(gt_labels, mask_inds)
    gt_anchor_labels = tf.where(mask, matched_labels, gt_anchor_labels)
    gt_anchor_bboxes = tf.where(mask, tf.gather(gt_bboxes, mask_inds),gt_anchor_bboxes)
    gt_anchor_scores = tf.reduce_max(jaccard, axis= 0)
#     mask_inds = tf.boolean_mask(mask_inds, mask)
#     gt_anchor_labels[mask] = gt_labels[mask_inds]
#     gt_anchor_bboxes[mask] = gt_bboxes[mask_inds]
#     gt_anchor_scores[mask] = jaccard[mask_inds, mask]
    
    
    #matching each ground truth box to the default box with the best jaccard overlap
    inds = tf.argmax(jaccard, axis = 1)
    gt_anchor_labels[1] = 0
    
#     sel_lables = tf.gather(gt_anchor_labels, inds)
#     sel_lables = gt_labels
    
#     gt_anchor_labels[inds] = gt_labels
    gt_anchor_bboxes[inds] = gt_bboxes
    gt_anchor_scores[inds] = jaccard[range(jaccard.shape[0]),inds]
    
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
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    
    
    gt_anchor_bboxes = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    
    
    
    
    return gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores







def test_anchor_matching():
    gt_bboxes = tf.constant([[0,0,1,2],[1,0,3,4],[100,100,105,102.5]])
    gt_labels = tf.constant([1,2,6])
    anchors = tf.constant([[100,100,105,105],[2,1,3,3.5],[0,0,10,10],[0.5,0.5,0.8,1.5]])
    
    jaccard = compute_jaccard(gt_bboxes, anchors)
    gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores = match_achors(gt_labels, gt_bboxes, anchors,jaccard,matching_threshold = 0.01)
    return gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores
with tf.Graph().as_default():
    
    r = test_anchor_matching()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        gt_anchor_labels, gt_anchor_bboxes,gt_anchor_scores = sess.run(r)
        print(gt_anchor_labels)
        
        
        
        

import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np

import math
from numpy import newaxis




class SSD():
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    def __init__(self):
        
#         self.anchor_size_bounds=[0.15, 0.90],
        #Configuration used to assign ground truth information to the model outputs that corresponds to all default boxes
        #the first element is the scale for this feature layer
        self.img_shape=(300, 300)
        self.num_classes=21
        self.no_annotation_label=21
        self.feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        self.feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.anchor_sizes=[(21., 45.),  #the first element is the scale for current layer, in this case, it's 21
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)]
       
        self.anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]]
        # the ration between input image size and feature layer size
        #it's used to map x and y of default box from feature layer to input layer
        #to determine the position of default boxes
        self.anchor_steps=[8, 16, 32, 64, 100, 300] 
        self.anchor_offset=0.5
        #Scaling of encoded coordinates.
        self.prior_scaling=[0.1, 0.1, 0.2, 0.2] 
        
        #normalization for conv4 3
        self.normalizations=[20, -1, -1, -1, -1, -1]
        
        #thresholding for ignoring "no annotation label"
        self.ignore_threshold = 0.5 
        
        # all of the computed anchors for this model, 
        # format: layer_number, numpy array format for x / y / w / h
        self.__anchors = None
        #format: layer number, numpy format for ymin,xmin,ymax,xmax
        self.__anchors_minmax = None
        
        return
    def get_all_anchors(self, minmaxformat=False):
        if self.__anchors is not None:
            if not minmaxformat:
                return self.__anchors
            else:
                if self.__anchors_minmax is not None:
                    return self.__anchors_minmax
                num_anchors = 0
                self.__anchors_minmax = []
                for i, anchors_layer in enumerate(self.__anchors):
                    anchors = np.zeros_like(anchors_layer)
                    cx = anchors_layer[...,0]
                    cy = anchors_layer[...,1]
                    w = anchors_layer[...,2]
                    h = anchors_layer[...,3]
                    anchors[..., 0] = cy - h / 2.
                    anchors[..., 1] = cx - w / 2.
                    anchors[..., 2] = cy + h / 2.
                    anchors[..., 3] = cx + w / 2. 
                    num_anchors = num_anchors + anchors.size
                    self.__anchors_minmax.append(anchors)
                print("Anchor numbers: {}".format(num_anchors))
                return self.__anchors_minmax
        anchors = self.ssd_anchors_all_layers()
        self.__anchors = []
        for _, anchors_layer in enumerate(anchors):
            yref, xref, href, wref = anchors_layer
            ymin = yref - href / 2.
            xmin = xref - wref / 2.
            ymax = yref + href / 2.
            xmax = xref + wref / 2.
            
            # Transform to center / size.
            cy = ((ymax + ymin) / 2.)[...,np.newaxis]
            cx = ((xmax + xmin) / 2.)[...,np.newaxis]
            h = (ymax - ymin)[...,np.newaxis]
            w = (xmax - xmin)[...,np.newaxis]
            temp_achors = np.concatenate([cx,cy,w,h], axis = -1)
           
            #append achors for this layer
            self.__anchors.append(temp_achors)
       
        return self.__anchors
    def ssd_bboxes_decode(self, feat_localizations,anchors):
        """Compute the relative bounding boxes from the layer features and
        corresponding reference anchor bounding boxes.
        Here we assume all the elements in feat_localizations are from the same layer, 
        otherwise the numpy slicing below won't work
    
        Return:
          numpy array Nx4: ymin, xmin, ymax, xmax
        """

        l_shape = feat_localizations.shape
        if feat_localizations.shape != anchors.shape:
            raise "feat_localizations and anchors should be of identical shape, and corresond to each other"
        
        # Reshape for easier broadcasting.
        feat_localizations = feat_localizations[np.newaxis,:]
        anchors = anchors[np.newaxis,:]
        
        xref = anchors[...,0]
        yref = anchors[...,1]
        wref = anchors[...,2]
        href = anchors[...,3]

    
        # Compute center, height and width
        cy = feat_localizations[..., 1] * href * self.prior_scaling[0] + yref
        cx = feat_localizations[..., 0] * wref * self.prior_scaling[1] + xref
        h = href * np.exp(feat_localizations[..., 3] * self.prior_scaling[2])
        w = wref * np.exp(feat_localizations[..., 2] * self.prior_scaling[3])
        
        # bboxes: ymin, xmin, xmax, ymax.
        bboxes = np.zeros_like(feat_localizations)
        bboxes[..., 0] = cy - h / 2.
        bboxes[..., 1] = cx - w / 2.
        bboxes[..., 2] = cy + h / 2.
        bboxes[..., 3] = cx + w / 2.
        bboxes = np.reshape(bboxes, l_shape)
        return bboxes
    def ssd_anchors_all_layers(self,
                           dtype=np.float32):
        """Compute anchor boxes for all feature layers.
        """
        layers_anchors = []
        for i, s in enumerate(self.feat_shapes):
            anchor_bboxes = self.__ssd_anchor_one_layer(self.img_shape, s,
                                                 self.anchor_sizes[i],
                                                 self.anchor_ratios[i],
                                                 self.anchor_steps[i],
                                                 offset=self.anchor_offset, dtype=dtype)
            layers_anchors.append(anchor_bboxes)
        return layers_anchors
    def __ssd_anchor_one_layer(self,img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
        """Computer SSD default anchor boxes for one feature layer.
    
        Determine the relative position grid of the centers, and the relative
        width and height.
    
        Arguments:
          feat_shape: Feature shape, used for computing relative position grids;
          size: Absolute reference sizes;
          ratios: Ratios to use on these features;
          img_shape: Image shape, used for computing height, width relatively to the
            former;
          offset: Grid offset.
    
        Return:
          y, x, h, w: Relative x and y grids, and height and width.
        """
        # Compute the position grid: simple way.
        # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        # y = (y.astype(dtype) + offset) / feat_shape[0]
        # x = (x.astype(dtype) + offset) / feat_shape[1]
        # Weird SSD-Caffe computation using steps values...
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) * step / img_shape[0]
        x = (x.astype(dtype) + offset) * step / img_shape[1]
    
        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)
    
        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.
        num_anchors = len(sizes) + len(ratios)
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        di = 1
        if len(sizes) > 1:
            h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
            w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
            di += 1
        for i, r in enumerate(ratios):
            h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        return y, x, h, w
    def __tf_ssd_bboxes_encode_layer(self, labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
        """Encode groundtruth labels and bounding boxes using SSD anchors from
        one layer.
    
        Arguments:
          labels: 1D Tensor(int64) containing groundtruth labels;
          bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
          anchors_layer: Numpy array with layer anchors;
          matching_threshold: Threshold for positive match with groundtruth bboxes;
          prior_scaling: Scaling of encoded coordinates.
    
        Return:
          (target_labels, target_localizations, target_scores): Target Tensors.
        """
        # Anchors coordinates and volume.
        yref, xref, href, wref = anchors_layer
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        vol_anchors = (xmax - xmin) * (ymax - ymin)
    
        # Initialize tensors...
        shape = (yref.shape[0], yref.shape[1], href.size)
        feat_labels = tf.zeros(shape, dtype=tf.int64)
        feat_scores = tf.zeros(shape, dtype=dtype)
    
        feat_ymin = tf.zeros(shape, dtype=dtype)
        feat_xmin = tf.zeros(shape, dtype=dtype)
        feat_ymax = tf.ones(shape, dtype=dtype)
        feat_xmax = tf.ones(shape, dtype=dtype)
    
        def jaccard_with_anchors(bbox):
            """Compute jaccard score between a box and the anchors.
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol \
                + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard
    
        def intersection_with_anchors(bbox):
            """Compute intersection between score a box and the anchors.
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            scores = tf.div(inter_vol, vol_anchors)
            return scores
    
        def condition(i, feat_labels, feat_scores,
                      feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """Condition: check label index.
            """
            r = tf.less(i, tf.shape(labels))
            return r[0]
    
        def body(i, feat_labels, feat_scores,
                 feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """Body: update feature labels, scores and bboxes.
            Follow the original SSD paper for that purpose:
              - assign values when jaccard > 0.5;
              - only update if beat the score of other bboxes.
            """
            # Jaccard score.
            label = labels[i]
            bbox = bboxes[i]
            jaccard = jaccard_with_anchors(bbox)
            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard, feat_scores) #jaccard is bigger than current matched bbox
            # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
            mask = tf.logical_and(mask, feat_scores > -0.5) #it's not "no annotations"
            mask = tf.logical_and(mask, label < num_classes) #the label value is valid
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype)
            # Update values using mask.
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = tf.where(mask, jaccard, feat_scores)
    
            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
    
            # Check no annotation label: ignore these anchors...
            # TODO, we probably can do without below code, will remove them in the future
            #This is because we've already checked the label previosly, which means feat_scores is already 0, 
            #thus belong to negative sample
            interscts = intersection_with_anchors(bbox)
            mask = tf.logical_and(interscts > ignore_threshold,
                                  label == no_annotation_label)
            # Replace scores by -1.
            feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)
    
            return [i+1, feat_labels, feat_scores,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]
        # Main loop definition.
        i = 0
        [i, feat_labels, feat_scores,
         feat_ymin, feat_xmin,
         feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                               [i, feat_labels, feat_scores,
                                                feat_ymin, feat_xmin,
                                                feat_ymax, feat_xmax])
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
        # Encode features.
        feat_cy = (feat_cy - yref) / href / prior_scaling[0]
        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
        feat_h = tf.log(feat_h / href) / prior_scaling[2]
        feat_w = tf.log(feat_w / wref) / prior_scaling[3]
        # Use SSD ordering: x / y / w / h instead of ours.
        feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
        return feat_labels, feat_localizations, feat_scores
    
    
    def tf_ssd_bboxes_encode(self, labels,
                             bboxes,
                             dtype=tf.float32,
                             scope='ssd_bboxes_encode'):
        """Encode groundtruth information for all default boxes, for one input image
    
        Arguments:
          labels: 1D Tensor(int64) containing groundtruth labels;
          bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
    
        Return:
          (target_labels, target_localizations, target_scores):
            Each element is a list of target Tensors.
            target_labels: target labels for all default boex,
            target_localizations: target localization offset for all default boxes
            target_scores: jaccard scores for all default boxes
            For default boxes that have no intersection with any of the ground truth boxes, target label and target score is 0,
            and target_localization is the whole input image
            If a default boxes intersect with multiple ground truth boxes, it will choose the one having the highest jaccard values
        """
        anchors = self.ssd_anchors_all_layers()
        with tf.name_scope(scope):
            target_labels = []
            target_localizations = []
            target_scores = []
            for i, anchors_layer in enumerate(anchors):
                with tf.name_scope('bboxes_encode_block_%i' % i):
                    t_labels, t_loc, t_scores = \
                        self.__tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                                   self.num_classes, self.no_annotation_label,
                                                   self.ignore_threshold,
                                                   self.prior_scaling, dtype)
                    target_labels.append(t_labels)
                    target_localizations.append(t_loc)
                    target_scores.append(t_scores)
            return target_labels, target_localizations, target_scores
   
    
    
    def run(self):
        
        
        return
    
    
g_ssd_model = SSD()

if __name__ == "__main__":   
    obj= SSD()
    obj.run()
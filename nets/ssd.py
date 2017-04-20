
import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np

import math
from numpy import newaxis
from nets import custom_layers
import tf_extended as tfe
from nets import ssd_common


class SSDModel():
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
        self.model_name = 'ssd_300_vgg'
        
        #post processing
        self.select_threshold = 0.01
        self.nms_threshold = 0.45
        self.select_top_k = 400
        self.keep_top_k = 200
        
        return
    def __additional_ssd_block(self, end_points, net):
        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        with tf.variable_scope("additional_blocks"):
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            end_points['block6'] = net
            # Block 7: 1x1 conv. Because the fuck.
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            end_points['block7'] = net 
            
            # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts)
            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))#may remvoe this layer later on
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = custom_layers.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            # Prediction and localisations layers.
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(self.feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = self.ssd_multibox_layer(end_points[layer],
                                              self.num_classes,
                                              self.anchor_sizes[i],
                                              self.anchor_ratios[i],
                                              self.normalizations[i])
                predictions.append(slim.softmax(p))
                logits.append(p)
                localisations.append(l)
    
        return predictions, localisations, logits, end_points
    def __arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Defines the VGG arg scope.
    
        Args:
          weight_decay: The l2 regularization coefficient.
    
        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as sc:
                    return sc

    
    def get_model(self,inputs, weight_decay=0.0005):
        # End_points collect relevant activations for external use.
        arg_scope = self.__arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            end_points = {}
            with tf.variable_scope('vgg_16', [inputs]):
                # Original VGG-16 blocks.
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                end_points['block1'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # Block 2.
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                end_points['block2'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # Block 3.
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                end_points['block3'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # Block 4.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                end_points['block4'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # Block 5.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                end_points['block5'] = net
                net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
                
                
        
            # Additional SSD blocks.
            # Block 6
            
            with tf.variable_scope(self.model_name):
                net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
                end_points['block6'] = net
                # Block 7: 1x1 conv. Because the fuck.
                net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
                end_points['block7'] = net
        
                # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
                end_point = 'block8'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                    net = custom_layers.pad2d(net, pad=(1, 1))
                    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block9'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = custom_layers.pad2d(net, pad=(1, 1))
                    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block10'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                end_points[end_point] = net
                end_point = 'block11'
                with tf.variable_scope(end_point):
                    net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
                end_points[end_point] = net
        
                # Prediction and localisations layers.
                predictions = []
                logits = []
                localisations = []
                for i, layer in enumerate(self.feat_layers):
                    with tf.variable_scope(layer + '_box'):
                        p, l = self.ssd_multibox_layer(end_points[layer],
                                                  self.num_classes,
                                                  self.anchor_sizes[i],
                                                  self.anchor_ratios[i],
                                                  self.normalizations[i])
                    predictions.append(slim.softmax(p))
                    logits.append(p)
                    localisations.append(l)
        
                return predictions, localisations, logits, end_points
    
    def ssd_multibox_layer(self, inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
        """Construct a multibox layer, return a class and localization predictions.
        """
        net = inputs
        if normalization > 0:
            net = custom_layers.l2_normalization(net, scaling=True)
        # Number of anchors.
        num_anchors = len(sizes) + len(ratios)
    
        # Location.
        num_loc_pred = num_anchors * 4
        loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                               scope='conv_loc')
        loc_pred = custom_layers.channel_to_last(loc_pred)
        loc_pred = tf.reshape(loc_pred,
                              self.tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
        # Class prediction.
        num_cls_pred = num_anchors * num_classes
        cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                               scope='conv_cls')
        cls_pred = custom_layers.channel_to_last(cls_pred)
        cls_pred = tf.reshape(cls_pred,
                              self.tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
        return cls_pred, loc_pred
    
    
    def tensor_shape(self, x, rank=3):
        """Returns the dimensions of a tensor.
        Args:
          image: A N-D Tensor of shape.
        Returns:
          A list of dimensions. Dimensions that are statically known are python
            integers,otherwise they are integer scalar tensors.
        """
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]
                

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
    def detected_bboxes(self, predictions, localisations,
                        clipping_bbox=None):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=self.select_threshold,
                                            num_classes=self.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=self.select_top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=self.nms_threshold,
                                 keep_top_k=self.keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes
    def decode_bboxes_all_ayers_tf(self, feat_localizations):
        """convert ssd boxes from relative to input image anchors to relative to input width/height
    
        Return:
          numpy array NlayersxNx4: ymin, xmin, ymax, xmax
        """
        anchors = self.ssd_anchors_all_layers()
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.prior_scaling)
        
    def decode_bboxes_layer(self, feat_localizations,anchors):
        """convert ssd boxes from relative to input image anchors to relative to 
        input width/height, for one signle feature layer
    
        Return:
          numpy array BatchesxHxWx4: ymin, xmin, ymax, xmax
        """

        l_shape = feat_localizations.shape
#         if feat_localizations.shape != anchors.shape:
#             raise "feat_localizations and anchors should be of identical shape, and corresond to each other"
        
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
    
    def get_losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
        """Loss functions for training the SSD 300 VGG network.
    
        This function defines the different loss components of the SSD, and
        adds them to the TF loss collection.
    
        Arguments:
          logits: (list of) predictions logits Tensors;
          localisations: (list of) localisations Tensors;
          gclasses: (list of) groundtruth labels Tensors;
          glocalisations: (list of) groundtruth localisations Tensors;
          gscores: (list of) groundtruth score Tensors;
        """
        with tf.name_scope(scope, 'ssd_losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []
            for i in range(len(logits)):
                dtype = logits[i].dtype
                with tf.name_scope('block_%i' % i):
                    # Determine weights Tensor.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(fpmask)
    
                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)
    
                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = slim.softmax(logits[i])
                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)
                    fnmask = tf.cast(nmask, dtype)
                    nvalues = tf.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)
                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)
    
                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    minval = val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, -nvalues > minval)
                    fnmask = tf.cast(nmask, dtype)
    
                    # Add cross-entropy loss.
                    with tf.name_scope('cross_entropy_pos'):
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=gclasses[i])
                        loss = tf.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)
    
                    with tf.name_scope('cross_entropy_neg'):
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                              labels=no_classes)
                        loss = tf.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)
    
                    # Add localization loss: smooth L1, L2, ...
                    with tf.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)
    
            # Additional total losses...
            with tf.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')
    
                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)
            model_loss = tf.add(total_loc, total_cross, 'model_loss')
            #Add regularziaton loss
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses,name='regularization_loss')
            total_loss = tf.add(model_loss, regularization_loss, 'total_loss')
        return total_loss
   
    
    
    def run(self):
        
        
        return
    
    
g_ssd_model = SSDModel()

if __name__ == "__main__":   
    obj= SSDModel()
    obj.run()
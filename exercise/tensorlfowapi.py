
import tensorflow as tf
import tensorflow.contrib.slim as slim


with tf.Graph().as_default():

#     i = tf.constant(0)
#     c = lambda i: tf.less(i, 10)
#     b = lambda i: tf.add(i, 1)
#     r = tf.while_loop(c, b, [i])
    
#     jaccard= tf.constant([0])
#     glabels = tf.constant([32],dtype=tf.int32)
#     rlabel = tf.constant([16])
#     jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)
#     jaccard = tf.Print(jaccard, [jaccard,glabels, rlabel], "in body")
    

    
    # Best fit, checking it's above threshold.
#     idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)

#     ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
    
    x = tf.constant(2)
    y = tf.constant(5)
    def f1(z): return x*y + z
    def f2(): return tf.add(y, 23)
    r = tf.cond(tf.equal(x < y, lambda: f1(1), f2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        res = sess.run(r)
        print(res)
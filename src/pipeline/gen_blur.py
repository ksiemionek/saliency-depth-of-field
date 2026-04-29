import numpy as np
import cv2
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from PyNET_Bokeh.model import PyNET


def load_model():
    x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 4])
    output_l1 = PyNET(x, instance_norm=True, instance_norm_level_1=False)[0]
    bokeh_img = tf.clip_by_value(output_l1, 0.0, 1.0)

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "PyNET_Bokeh/models/original/pynet_bokeh_level_0")

    return x, bokeh_img, sess


def generate_blur(image, depth, model):
    x, bokeh_img, sess = model

    h, w = image.shape[:2]
    I = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)
    I_depth = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

    new_height = (I.shape[0] // 32) * 32
    new_width = (I.shape[1] // 32) * 32
    I = I[:new_height, :new_width, :]
    I_depth = I_depth[:new_height, :new_width]

    I_temp = np.zeros((I.shape[0], I.shape[1], 4))
    I_temp[:, :, 0:3] = I
    I_temp[:, :, 3] = I_depth

    I = np.float32(I_temp) / 255.0
    I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

    bokeh_tensor = sess.run(bokeh_img, feed_dict={x : I})
    bokeh_image = np.reshape(bokeh_tensor, [I.shape[1] * 2, I.shape[2] * 2, 3])

    return (bokeh_image * 255).astype(np.uint8)

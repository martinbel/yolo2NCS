import tensorflow as tf
import cv2
import numpy as np
from ncsmodel.DarknetReferenceNet import Net


NN = Net(load_weights=True)
cv2_image = cv2.imread("images/person.jpg", 0)
image = NN.image

cv2_image = np.expand_dims(cv2_image, 2)
resized_image, image_data = NN.preprocess_image(cv2_image)
image_data = np.expand_dims(image_data, 3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prediction = sess.run(NN.predict(), feed_dict={image: image_data}) # 200 ms
    # Save Network
    saver = tf.train.Saver()
    saver.save(sess, "ncs_darknetreference/NN.ckpt")
    tf.train.write_graph(sess.graph_def, "ncs_darknetreference", "NN.pb", as_text=False)
    output_image, boxes = NN.postprocess(prediction, resized_image, 0.5, 0.5) # 32 ms
    cv2.imshow('image', output_image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

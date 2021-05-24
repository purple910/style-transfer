# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to paintA.')
tf.app.flags.DEFINE_string("model_file", "models/vgg16_scream/fast_style_transfer.ckpt-done", "")
# tf.app.flags.DEFINE_string("model_file", "E:\\python\\demo\\图像风格迁移\\vgg16_paintA_denoised_starry\\fast-style-model.ckpt-done.meta", "")
# tf.app.flags.DEFINE_string("model_file", 'model/vgg16_wave.ckpt-done', "")
tf.app.flags.DEFINE_string("image_file", "content/chicago.jpg", "")
tf.app.flags.DEFINE_string("out_file", "result/res_stu2.jpg", "")

FLAGS = tf.app.flags.FLAGS


def main(flags):
    # Get image's height and width.
    height = 0
    width = 0
    with open(flags.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if flags.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                flags.loss_model,
                is_training=False)
            image = reader.get_image(flags.image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            flags.model_file = os.path.abspath(flags.model_file)
            saver.restore(sess, flags.model_file)

            # Generate and write image data to file.
            with open(flags.out_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                tf.logging.info('Done. Please check %s.' % flags.out_file)


if __name__ == '__main__':
    main(FLAGS)

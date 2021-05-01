from os import listdir
from os.path import join, isfile

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import time
import os

slim = tf.contrib.slim

train_image_path = 'E:\\python\\train2014'
train_model_name = 'fast_style_transfer.ckpt'
train_model_name_done = 'fast_style_transfer.ckpt-done'
style_image = "style/jpg/picasso.jpg"
model_path = "models"
naming = "vgg16_picasso"
loss_model_file = "vggnet/vgg_16.ckpt"
loss_model = "vgg_16"
content_layers = ["vgg_16/conv3/conv3_3"]
style_layers = ["vgg_16/conv1/conv1_2",
                "vgg_16/conv2/conv2_2",
                "vgg_16/conv3/conv3_3",
                "vgg_16/conv4/conv4_3"]
checkpoint_exclude_scopes = "vgg_16/fc"
content_weight = 1.0
style_weight = 50.0
tv_weight = 0.0
image_size = 256
batch_size = 4
epoch = 2


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def _get_init_fn():
    """
    此功能是从TF slim复制而来的。
    返回首席工作人员运行的功能以热启动培训。
    请注意，仅在第一个全局步骤中初始化模型时才运行init_fn。
    返回：由主管运行的初始化函数。
    """
    tf.logging.info('Use pretrained model %s' % loss_model_file)

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


def get_style_features(training_path):
    """
    对于“ style_image”，预处理步骤为：
    1. 将短边调整为FLAGS.image_size
    2. 应用中央作物
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            loss_model,
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            loss_model,
            is_training=False)

        # Get the style image data
        size = image_size
        img_bytes = tf.read_file(style_image)
        if style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        # image = _aspect_preserving_resize(image, size)

        # Add the batch dimension
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        # images = tf.stack([image_preprocessing_fn(image, size, size)])

        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)

        with tf.Session() as sess:
            # Restore variables for loss network.
            init_func = _get_init_fn()
            init_func(sess)

            # Make sure the 'generated' directory is exists.
            # if os.path.exists('generated') is False:
            #     os.makedirs('generated')
            # Indicate cropped style image path
            save_file = training_path + '/target_style_' + naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)


def image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess_fn(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def relu(input):
    relu = tf.nn.relu(input)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))
    return nan_to_zero


def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)
        residual = x + conv2
        return residual


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel / 2), int(kernel / 2)], [int(kernel / 2), int(kernel / 2)], [0, 0]],
                          mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def net(image, training):
    # 在通过前稍微填充一下时，边框效果较小..
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    # print(res5.get_shape())
    with tf.variable_scope('deconv1'):
        # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        # deconv_test = relu(instance_norm(conv2d(deconv2, 32, 32, 2, 1)))
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    y = (deconv3 + 1) * 127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y


def style_loss_fun(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary


def content_loss_fun(endpoints_dict, content_layers):
    loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(
            size)  # remain the same as in the paper
    return loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0],
                                                                                     [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0],
                                                                                    [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss


def main():
    # 确保训练路径存在.
    training_path = os.path.join(model_path, naming)
    if not (os.path.exists(training_path)):
        os.makedirs(training_path)

    style_features_t = get_style_features(training_path)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            """建立网络"""
            network_fn = nets_factory.get_network_fn(
                loss_model,
                num_classes=1,
                is_training=False)

            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                loss_model,
                is_training=False)
            processed_images = image(batch_size, image_size, image_size,
                                     train_image_path, image_preprocessing_fn, epochs=epoch)
            generated = net(processed_images, training=True)
            processed_generated = [image_preprocessing_fn(image, image_size, image_size)
                                   for image in tf.unstack(generated, axis=0, num=batch_size)
                                   ]
            processed_generated = tf.stack(processed_generated)
            _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)

            # 记录丢失网络的结构
            tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
            for key in endpoints_dict:
                tf.logging.info(key)

            """建筑损失"""
            content_loss = content_loss_fun(endpoints_dict, content_layers)
            style_loss, style_loss_summary = style_loss_fun(endpoints_dict, style_features_t, style_layers)
            tv_loss = total_variation_loss(generated)  # use the unprocessed image

            loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tv_loss

            """准备训练"""
            global_step = tf.Variable(0, name="global_step", trainable=False)

            variable_to_train = []
            for variable in tf.trainable_variables():
                if not (variable.name.startswith(loss_model)):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

            variables_to_restore = []
            for v in tf.global_variables():
                if not (v.name.startswith(loss_model)):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # 恢复损失网络的变量.
            init_func = _get_init_fn()
            init_func(sess)

            # 如果检查点文件存在，则恢复训练模型的变量.
            last_file = tf.train.latest_checkpoint(training_path)
            if last_file:
                tf.logging.info('Restoring model from {}'.format(last_file))
                saver.restore(sess, last_file)

            """开始训练"""
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    # 获取的loss就是train_op执行完之后的loss了
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    if step % 10 == 0:
                        tf.logging.info('step: %d,  total Loss %f, secs/step: %f' % (step, loss_t, elapsed_time))
                    """checkpoint"""
                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(training_path, train_model_name), global_step=step)

            except tf.errors.OutOfRangeError:
                saver.save(sess, os.path.join(training_path, train_model_name_done))
                tf.logging.info('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()

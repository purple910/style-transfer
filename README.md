
#### conf 训练模型的基本配置文件
#### content 内容图片
#### data 数据集文件
#### model 训练好的模型
#### models 自己训练的模型
#### nets preprocessing 复制TensorFlow Slim项目的原始文件.定义了一些ImageNet预训练模型. 
#### result 输出图片
#### style 风格图片,分成jpg与png
#### train 训练的图片
#### vggnet 预训练模型

## VGG16 vgg_16.ckpt
##### eval.py 用于使用训练好的模型进行图像的快速风格转移. (ok)
##### export.py 将模板导出为.pb文件.(可以不用在意) 
##### losses.py 用于定义风格损失,内容损失.
##### model.py 用于定义图像生成网络.
##### reader.py I/O接口.将训练的图片读入TensorFlow.
##### train_vgg16.py 用于训练模型.
##### utils.py 定义其他一些方便的函数.


## VGG16 vgg16.npy 
##### backward.py 训练模型
##### forward.py 用于定义风格损失,内容损失.
##### generateds.py 生成数据集
##### test.py 测试多风格融合 (ok)



## VGG19 imagenet-vgg-verydeep-19.mat
##### train_vgg19.py 训练模型
##### main.py 用于使用自己训练好的模型进行风格转移.  (ok)



## npy == numpy np.load()
## ckpt == checkpoint,ChecK PoinT (slim = tf.contrib.slim)
## mat == scipy.io.loadmat()

## 1.单风格变换
## 2.预训练模型对比
## 3.多风格融合对比

| 1    | checkpoint | 文本文件,记录最新的模型文件列表    |
| ---- | ---------- | ---------------------------------- |
| 2    | .data      | 包含训练变量的值value              |
| 3    | .index     | 包含.data和.meta文件对应关系       |
| 4    | .meta      | 包含网络图结构,如GraphDef,SaverDef |
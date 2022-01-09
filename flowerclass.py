#开发环境 tensorflow core 2.1，需下载数据集flowerclass或者自制数据集
import tensorflow as tf
import pathlib
import random
data_dir = "E:/pycharm/tflite1/class3/flowerclass"
AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root = pathlib.Path(data_dir)
print(data_root)
for item in data_root.iterdir():
  print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
print(all_image_paths)
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

print(all_image_paths[:10])


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

"""检查图片
现在让我们快速浏览几张图片，这样你知道你在处理什么："""
import os
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)
print("attributions is ",attributions)
import IPython.display as display
import matplotlib.pyplot as plt
def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
for n in range(3):
  image_path = random.choice(all_image_paths)
  image = plt.imread(image_path)
  plt.imshow(image)
  plt.show()

"""加载和格式化图片
TensorFlow 包含加载和处理图片时你需要的所有工具："""
img_path = all_image_paths[0]
print(img_path)
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")
"""将它解码为图像 tensor（张量）："""
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
"""根据你的模型调整其大小："""
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0  #归一化
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

"""将这些包装在一个简单的函数里，以备后用。"""
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image
def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)
import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.title(label_names[label].title())
plt.show()

"""构建一个 tf.data.Dataset
一个图片数据集
构建 tf.data.Dataset 最简单的方法就是使用 from_tensor_slices 方法。

将字符串数组切片，得到一个字符串数据集："""
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)

"""现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片。"""
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.show()

"""一个(图片, 标签)对数据集
使用同样的 from_tensor_slices 方法你可以创建一个标签数据集："""
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
  print(label_names[label.numpy()])

"""由于这些数据集顺序相同，你可以将他们打包在一起得到一个(图片, 标签)对数据集："""
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
"""注意：当你拥有形似 all_image_labels 和 all_image_paths 的数组，tf.data.dataset.Dataset.zip 的替代方法是将这对数组切片。"""
# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

"""训练的基本方法
要使用此数据集训练模型，你将会想要数据：

被充分打乱。
被分割为 batch。
永远重复。
尽快提供 batch。
使用 tf.data api 可以轻松添加这些功能。"""
BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

"""这里有一些注意事项：

顺序很重要。

在 .repeat 之后 .shuffle，会在 epoch 之间打乱数据（当有些数据出现两次的时候，其他数据还没有出现过）。

在 .batch 之后 .shuffle，会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。

你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。

在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。

在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。

最后一点可以通过使用 tf.data.Dataset.apply 方法和融合过的 tf.data.experimental.shuffle_and_repeat 函数来解决:
"""
"""ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)"""
"""传递数据集至模型,迁移学习transfer-learning
从 tf.keras.applications 取得 MobileNet v2 副本。

该模型副本会被用于一个简单的迁移学习例子。

设置 MobileNet 的权重为不可训练："""
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False
"""该模型期望它的输出被标准化至 [-1,1] 范围内："""
"""在你将输出传递给 MobilNet 模型之前，你需要将其范围从 [0,1] 转化为 [-1,1]："""
#image identity [0,1] 转化为 [-1,1]：
def change_range(image,label):
  return 2*image-1, label
keras_ds = ds.map(change_range)
"""MobileNet 为每张图片的特征返回一个 6x6 的空间网格。

传递一个 batch 的图片给它，查看结果："""
# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

"""构建一个包装了 MobileNet 的模型并在 tf.keras.layers.Dense 输出层之前使用 tf.keras.layers.GlobalAveragePooling2D 来平均那些空间向量：
"""
model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation = 'softmax')])
"""现在它产出符合预期 shape(维数)的输出："""
logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

"""编译模型以描述训练过程："""
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
"""此处有两个可训练的变量 —— Dense 层中的 weights（权重） 和 bias（偏差）："""
print(len(model.trainable_variables))
model.summary()
"""你已经准备好来训练模型了。

注意，出于演示目的每一个 epoch 中你将只运行 3 step，但一般来说在传递给 model.fit() 之前你会指定 step 的真实数量，如下所示："""
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print(steps_per_epoch)
model.fit(ds, epochs=5, steps_per_epoch=steps_per_epoch)

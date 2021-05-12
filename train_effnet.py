import os
import tensorflow as tf
from tensorflow.keras import backend as K
try:
    from kaggle_datasets import KaggleDatasets
except ImportError:
    pass

try:
    from util import build_image_model, seed_everything
except ImportError:
    pass

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


class CFG:
    EPOCHS = [30]  # 30
    BATCH_SIZE = 128
    IMAGE_SIZE = [512, 512]
    SEED = 42
    LR = 0.001
    AUTO = tf.data.experimental.AUTOTUNE
    input_dir = '../input'
    output_dir = 'drive/MyDrive/Colab Notebooks/shopee-product-matching/data/output/effnet/20210510'
    holdout = False


def makedir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def arcface_format(posting_id, image, label_group):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group


# Data augmentation function
def data_augment(posting_id, image, label_group):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return posting_id, image, label_group


# Function to decode our images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, CFG.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "posting_id": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label_group": tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['posting_id']
    image = decode_image(example['image'])
    label_group = tf.cast(example['label_group'], tf.int32)
    return posting_id, image, label_group


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=CFG.AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=CFG.AUTO)
    return dataset


# This function is to get our training tensors
def get_training_dataset(filenames, ordered=False):
    dataset = load_dataset(filenames, ordered=ordered)
    dataset = dataset.map(data_augment, num_parallel_calls=CFG.AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=CFG.AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(CFG.AUTO)
    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, ordered=True):
    dataset = load_dataset(filenames, ordered=ordered)
    dataset = dataset.map(arcface_format, num_parallel_calls=CFG.AUTO)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    dataset = dataset.prefetch(CFG.AUTO)
    return dataset


# Function for a custom learning rate scheduler with warmup and decay
def get_lr_callback():
    lr_start = 0.000001
    lr_max = 0.000005 * CFG.BATCH_SIZE
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


def run(model_name='EfficientNetB3'):
    seed_everything(CFG.SEED)
    makedir(CFG.output_dir)
    if CFG.holdout:
        n_classes = 8811
        train_data_size = 27400
        train = ['gs://kds-7a96af6666c458951a3b871a643c85f34bcf7947bfc6142fcf360c9b/train.tfrec']
    else:
        n_classes = 11014
        train_data_size = 34250
        train = ['gs://kds-5ba192632c3240653c3b3a0a12595f0821ceb3937a8d7ea4a1ed343e/train.tfrec']
    train_dataset = get_training_dataset(train, ordered=False)
    train_dataset = train_dataset.map(lambda posting_id, image, label_group: (image, label_group))

    STEPS_PER_EPOCH = train_data_size // CFG.BATCH_SIZE
    K.clear_session()

    for epoch in CFG.EPOCHS:
        print(f'epoch={epoch} start')
        with strategy.scope():
            model = build_image_model(n_classes=n_classes, image_size=CFG.IMAGE_SIZE, weights='imagenet', model_name=model_name)
            opt = tf.keras.optimizers.Adam(learning_rate=CFG.LR)
            model.compile(
                optimizer=opt,
                loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
        if CFG.holdout:
            output_file_name = f'{model_name}_epoch{epoch}_holdout.h5'
        else:
            output_file_name = f'{model_name}_epoch{epoch}.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(CFG.output_dir, output_file_name), monitor='loss', verbose=2, save_best_only=True, save_weights_only=True, mode='min')
        model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=epoch, callbacks=[checkpoint, get_lr_callback()], verbose=2)

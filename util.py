import math
import numpy as np
import tensorflow as tf
import gc
import random
import os
import matplotlib.pyplot as plt
import cv2
import re
try:
    import efficientnet.tfkeras as efn
except ImportError:
    pass


class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    '''

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores


def f1_precision_recall(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x))
    y_pred = y_pred.apply(lambda x: set(x))

    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = y_pred.apply(lambda x: len(x)).values - tp
    fn = y_true.apply(lambda x: len(x)).values - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1, precision, recall


def build_image_model(n_classes, image_size, model_name, weights=None):
    margin = ArcMarginProduct(
        n_classes=n_classes,
        s=30,
        m=0.7,
        name='head/arc_margin',
        dtype='float32'
    )
    inp = tf.keras.layers.Input(shape=(*image_size, 3), name='inp1')
    label = tf.keras.layers.Input(shape=(), name='inp2')
    if model_name == 'EfficientNetB3':
        x = efn.EfficientNetB3(weights=weights, include_top=False)(inp)
    elif model_name == 'EfficientNetB4':
        x = efn.EfficientNetB4(weights=weights, include_top=False)(inp)
    elif model_name == 'EfficientNetB5':
        x = efn.EfficientNetB5(weights=weights, include_top=False)(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = margin([x, label])
    output = tf.keras.layers.Softmax(dtype='float32')(x)
    model = tf.keras.models.Model(inputs=[inp, label], outputs=[output])
    return model


def read_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    IMAGE_SIZE = [512, 512]
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def get_image_embeddings(model, image_paths, batch_size, chunk=5000):
    embeds = []
    iterator = np.arange(np.ceil(len(image_paths) / chunk))
    for j in iterator:
        a = int(j * chunk)
        b = int((j + 1) * chunk)

        AUTO = tf.data.experimental.AUTOTUNE
        dataset = tf.data.Dataset.from_tensor_slices(image_paths[a:b])
        dataset = dataset.map(read_image, num_parallel_calls=AUTO)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTO)

        # predict
        image_embeddings = model.predict(dataset)
        embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    del embeds
    gc.collect()
    return image_embeddings


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def plot_images(dataframe, column_name, value):
    plt.figure(figsize=(30, 30))
    value_filter = dataframe[dataframe[column_name] == value]
    image_paths = value_filter['image_path'].to_list()
    print(f'Total images: {len(image_paths)}')
    posting_id = dataframe['posting_id'].to_list()
    for i, j in enumerate(zip(image_paths, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(j[0]), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)


def preprocess_title(df):
    # emoji
    RE_EMOJI = re.compile(r"\\x[A-Za-z0-9]{2}", flags=re.UNICODE)

    def strip_emoji(text):
        return RE_EMOJI.sub(r'', text)

    df['title'] = df['title'].apply(lambda x: strip_emoji(x))

    # lower
    df['title'] = df['title'].str.lower()

    df['title'].replace(regex={
        '^b"': '"'
    }, inplace=True)

    # punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    punctuation = r"""!"#$'()*+,-/:;<=>?@[\]^_`{|}~"""
    df['title'] = df['title'].apply(lambda s: s.translate(str.maketrans(punctuation, ' ' * len(punctuation))))

    # stop words
    STOP_WORDS = [
        'bisa cod',
        'cod',  # 配送
        'bayar di tempat',  # 代引き
        'gogomart',  # スーパーマーケット
        'ready stock',
        'ready',
        'official',
        'free',
        '100% original',
        '100% ori',
        '100%',
        'original',
        'in stock',
        'ready stock',
        'jakarta',
        'hot deal',
        'bestdeal',
        'best deal',
        'deal',
        'bayar ditempat',
        'pembayaran',
        'bayar',
        'best seller',
        'bestseller',
        'top seller',
        'reseller',
        'seller',
        'flash sale',
        'promo',
        'hot sale',
        'great sale',
        'sale',
        'new',
        'high quality',
        'good quality',
        'premium quality',
        'best quality',
        'top quality',
        'highquality'
    ]
    for w in STOP_WORDS:
        df['title'].replace(regex={w: ''}, inplace=True)

    df['title'].replace(regex={
        # '%': 'percent',
        r'(\d+)\.(\d+)': r'\1_\2',
        r'(\d+)([A-Za-z]+)': r'\1 \2',
    }, inplace=True)

    df['title'].replace(regex={
        r'(\d+)\s(gr|g|gram|grm)($|\s+)': r'\1gram\3',
        r'(\d+)\s(kg|percent|micron|ply)($|\s+)': r'\1\2\3',
        r'(\d+)\s(mm|ml|cm)($|\s+)': r'\1\2\3',
        r'(\d+)\s(mm|ml|cm)x': r'\1\2 x',
        r'x(\d+)\s(mm|ml|cm)($|\s+)': r'x \1\2\3',
        r'(s|m|l|xl|xxl)(\d+)($|\s+)': r'\1 \2 ',
        r'(\d+)\s(pc|pcs)($|\s+)': r'\1pc\3',
        r'(\d+)in(\d+)': r'\1 in \2',
        r'(\d+)x(\d+)': r'\1 x \2',
        r'\.+$': '',
        r'\s+': ' ',
        r'\s+$': '',
        r'\w+\.com': '',
        r'([A-Za-z]+)(\d+)': r'\1 \2',
        r'(\w+)&(\w+)': r'\1 & \2',
        r'(\d+)x ': r'\1 x ',
    }, inplace=True)
    return df
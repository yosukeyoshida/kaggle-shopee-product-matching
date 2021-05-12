import numpy as np
try:
    import efficientnet.tfkeras as efn
except ImportError:
    pass
import pandas as pd
import tensorflow as tf
import warnings
warnings.simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
import os
try:
    from util import preprocess_title, seed_everything, ArcMarginProduct
except ImportError:
    pass


class CFG:
    EPOCHS = 25
    BATCH_SIZE = 16
    SEED = 123
    LR = 0.00001
    input_dir = 'drive/MyDrive/Colab Notebooks/shopee-product-matching/data/input'
    train_file_path = os.path.join(input_dir, 'shopee-product-matching', 'train.csv')
    train_holdout_file_path = os.path.join(input_dir, 'shopee-product-matching', 'train_holdout.csv')
    tokenization_input_file_path = os.path.join(input_dir, 'tokenization', 'tokenization.py')
    # tokenization_output_file_path = '../working/tokenization.py'
    tokenization_output_file_path = './tokenization.py'
    bert_model_dir = os.path.join(input_dir, 'bert-en-uncased-l24-h1024-a16-1')
    output_dir = 'drive/MyDrive/Colab Notebooks/shopee-product-matching/data/output/bert/20210504'
    holdout = False


from shutil import copyfile
copyfile(src=CFG.tokenization_input_file_path, dst=CFG.tokenization_output_file_path)
import tokenization


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_bert_model(bert_layer, n_classes, max_len=512):
    margin = ArcMarginProduct(
        n_classes=n_classes,
        s=30,
        m=0.5,
        name='head/arc_margin',
        dtype='float32'
    )

    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    label = tf.keras.layers.Input(shape=(), name='label')

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    x = margin([clf_output, label])
    output = tf.keras.layers.Softmax(dtype='float32')(x)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids, label], outputs=[output])
    return model


def run_train(x_train, y_train, n_classes):
    bert_layer = hub.KerasLayer(CFG.bert_model_dir, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    x_train = bert_encode(x_train['title'].values, tokenizer, max_len=70)
    y_train = y_train.values
    x_train = (x_train[0], x_train[1], x_train[2], y_train)
    bert_model = build_bert_model(bert_layer, n_classes=n_classes, max_len=70)
    bert_model.compile(optimizer=tf.keras.optimizers.Adam(lr=CFG.LR), loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    if CFG.holdout:
        output_file_name = f'bert_holdout_epoch{CFG.EPOCHS}.h5'
    else:
        output_file_name = f'bert_epoch{CFG.EPOCHS}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(CFG.output_dir, output_file_name), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    bert_model.fit(x_train, y_train, epochs=CFG.EPOCHS, callbacks=[checkpoint], batch_size=CFG.BATCH_SIZE, verbose=1)


def makedir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def run():
    seed_everything(CFG.SEED)
    makedir(CFG.output_dir)
    if CFG.holdout:
        n_classes = 8811
        data = pd.read_csv(CFG.train_holdout_file_path)
        data = data[data['fold'] == 0]
        data = data.drop(['fold'], axis=1)
    else:
        n_classes = 11014
        data = pd.read_csv(CFG.train_file_path)
    data = preprocess_title(data)
    encoder = LabelEncoder()
    data['label_group'] = encoder.fit_transform(data['label_group'])
    x_train = data[['title']]
    y_train = data['label_group']
    run_train(x_train, y_train, n_classes)
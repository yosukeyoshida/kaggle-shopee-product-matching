import os
import numpy as np
import pandas as pd
import gc
import cudf
import cupy
from cuml.neighbors import NearestNeighbors
from cuml.feature_extraction.text import TfidfVectorizer
from collections import Counter
import tensorflow as tf
try:
    from util import f1_score, build_image_model, get_image_embeddings, f1_precision_recall, preprocess_title
except ImportError:
    pass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import transformers

from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import math
import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import timm


class CFG:
    BATCH_SIZE = 8
    IMAGE_SIZE = [512, 512]

    GET_CV = True
    holdout = True
    debug = False

    input_dir = '../input'
    train_file_path = os.path.join(input_dir, 'shopee-product-matching', 'train.csv')
    train_holdout_file_path = os.path.join(input_dir, 'shopee-holdout-tfrecord', 'train_holdout.csv')
    test_file_path = os.path.join(input_dir, 'shopee-product-matching', 'test.csv')
    train_images_dir = os.path.join(input_dir, 'shopee-product-matching', 'train_images')
    test_images_dir = os.path.join(input_dir, 'shopee-product-matching', 'test_images')
    tfidf_save_path = 'tfidf.csv'
    transformer_model_path = os.path.join(input_dir, 'sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer')
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(transformer_model_path)

    image_preds_strict_column = 'image_preds_strict'
    image_preds_loose_column = 'image_preds_loose'
    b4_image_preds_strict_column = 'b4_image_preds_strict'
    b4_image_preds_loose_column = 'b4_image_preds_loose'
    tfidf_preds_strict_column = 'tfidf_preds_strict'
    tfidf_preds_loose_column = 'tfidf_preds_loose'
    sbert_preds_strict_column = 'sbert_preds_strict'
    sbert_preds_loose_column = 'sbert_preds_loose'

    image_threshold_strict = 0.3
    b4_image_threshold_strict = 0.3
    # nfnet_threshold_strict = 0.3
    tfidf_threshold_strict = 0.85
    sbert_threshold_strict = 0.85

    if holdout:
        effnetb3_weight_path = os.path.join(input_dir, '20210421-holdout/EfficientNetB3_epoch30_holdout.h5')
        effnetb4_weight_path = os.path.join(input_dir, '20210423-holdout/EfficientNetB4_epoch30_holdout.h5')
        sbert_model_path = os.path.join(input_dir, '20210421-holdout/sentence_transfomer_xlm_best_loss_num_epochs_25_arcface_holdout.bin')
        nfnet_weight_path = os.path.join(input_dir, '20210505-nfnet/arcface_512x512_nfnet_l0_mish_holdout.pt')
    else:
        effnetb3_weight_path = os.path.join(input_dir, '20210411-effnet/20210411/EfficientNetB3.h5')
        effnetb4_weight_path = os.path.join(input_dir, '20210423-effnetb4/EfficientNetB4_epoch30.h5')
        sbert_model_path = os.path.join(input_dir, '20210421-sbert/sentence_transfomer_xlm_best_loss_num_epochs_25_arcface.bin')
        # nfnet_weight_path = os.path.join(input_dir, 'shopee-pytorch-models/arcface_512x512_nfnet_l0 (mish).pt')
        nfnet_weight_path = os.path.join(input_dir, '20210505-nfnet/arcface_512x512_nfnet_l0_mish.pt')


def tf_mem_limit(limit=1):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)


def read_dataset():
    if CFG.GET_CV:
        if CFG.holdout:
            df = pd.read_csv(CFG.train_holdout_file_path)
            df = df[df['fold'] == 1]
            df = df.drop(['fold'], axis=1)
        else:
            df = pd.read_csv(CFG.train_file_path)
        if CFG.debug:
            df = df.iloc[:100, :]
        tmp = df.groupby('label_group')['posting_id'].agg('unique').to_dict()
        df['target'] = df['label_group'].map(tmp)
        image_dir = CFG.train_images_dir
    else:
        df = pd.read_csv(CFG.test_file_path)
        image_dir = CFG.test_images_dir
    image_paths = df['image'].apply(lambda x: os.path.join(image_dir, x))

    preprocess_title(df)
    return df, image_paths


def get_tfidf_embeddings(df):
    df_gf = cudf.DataFrame(df)
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=25_000)
    text_embeddings = model.fit_transform(df_gf.title).toarray()
    # dic = model.get_feature_names().to_array()
    # np.savetxt(CFG.tfidf_save_path, dic, fmt='%s')
    del model
    gc.collect()
    return text_embeddings


def get_text_preds(df, embeddings, threshold_strict):
    preds_strict = []
    preds_loose = []
    CHUNK = 1024 * 4
    CTS = len(df) // CHUNK
    if len(df) % CHUNK != 0:
        CTS += 1
    for j in range(CTS):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(df))
        cts = cupy.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            o = df.iloc[cupy.asnumpy(cupy.where(cts[k, ] > threshold_strict)[0])].posting_id.values
            preds_strict.append(o)
            if len(o) >= 2:
                preds_loose.append([])
            else:
                _threshold = threshold_strict
                _preds_loose = []
                while _threshold > 0.5 and len(_preds_loose) < 2:
                    _threshold -= 0.02
                    _preds_loose = df.iloc[cupy.asnumpy(cupy.where(cts[k, ] > _threshold)[0])].posting_id.values
                preds_loose.append(_preds_loose)
    return preds_strict, preds_loose


def get_neighbors(df, embeddings, threshold_strict, KNN=50, metric='cosine'):
    if len(df) == 3:
        KNN = 2
    model = NearestNeighbors(n_neighbors=KNN, metric=metric)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    preds_strict = []
    preds_loose = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k, ] < threshold_strict)[0]
        preds_strict.append(df['posting_id'].iloc[indices[k, idx]].values)
        if len(idx) >= 2:
            preds_loose.append([])
        else:
            _threshold = threshold_strict
            _preds_loose = []
            while _threshold < 0.5 and len(_preds_loose) < 2:
                _threshold += 0.02
                idx = np.where(distances[k, ] < _threshold)[0]
                _preds_loose = df['posting_id'].iloc[indices[k, idx]].values
            preds_loose.append(_preds_loose)
    del model, distances, indices
    gc.collect()
    return preds_strict, preds_loose


def get_single_image_embeddings(image_paths, n_classes, model_name, effnet_weight_path):
    print(f'effnet_weight_path={effnet_weight_path}')
    model = build_image_model(n_classes=n_classes, image_size=CFG.IMAGE_SIZE, model_name=model_name, weights=None)
    model.load_weights(effnet_weight_path)
    model = tf.keras.models.Model(inputs=model.input[0], outputs=model.layers[-4].output)
    image_embeddings = get_image_embeddings(model=model, image_paths=image_paths, batch_size=CFG.BATCH_SIZE)
    del model
    gc.collect()
    return image_embeddings


class ShopeeDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv.reset_index()

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        text = row['title']
        text = CFG.TOKENIZER(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        return input_ids, attention_mask


class ShopeeNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='bert-base-uncased',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(model_name)
        final_in_features = self.transformer.config.hidden_size

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids, attention_mask):
        feature = self.extract_feat(input_ids, attention_mask)
        return F.normalize(feature)

    def extract_feat(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        features = x[0]
        features = features[:, 0, :]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)

        return features


def get_sbert_embeddings(df, sbert_model_path=CFG.sbert_model_path):
    print(f'sbert_model_path={sbert_model_path}')
    embeds = []

    model_params = {
        'n_classes': 11014,
        'model_name': CFG.transformer_model_path,
    }

    model = ShopeeNet(**model_params)
    model.eval()

    model.load_state_dict(dict(list(torch.load(sbert_model_path).items())[:-1]))
    device = torch.device('cuda')
    model = model.to(device)

    text_dataset = ShopeeDataset(df)
    text_loader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=16,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    with torch.no_grad():
        for input_ids, attention_mask in text_loader:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            feat = model(input_ids, attention_mask)
            text_embeddings = feat.detach().cpu().numpy()
            embeds.append(text_embeddings)

    del model
    text_embeddings = np.concatenate(embeds)
    del embeds
    gc.collect()
    return text_embeddings


def replace_single_preds(x):
    if len(x['preds']) == 1:
        if len(x['preds_loose_union']) == 2:
            return x['preds_loose_union']
        else:
            return x['preds']
    else:
        return x['preds']


def get_test_transforms():

    return A.Compose(
        [
            A.Resize(512, 512, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


class ShopeeNfnetDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(1)


class ArcMarginProductNfnet(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProductNfnet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class ShopeeModelNfnet(nn.Module):

    def __init__(
            self,
            n_classes,
            model_name='eca_nfnet_l0',
            fc_dim=512,
            margin=0.5,
            scale=30,
            use_fc=True,
            pretrained=False):

        super(ShopeeModelNfnet, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'efficientnet_b3':
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'tf_efficientnet_b5_ns':
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'eca_nfnet_l0':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProductNfnet(
            final_in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        # logits = self.final(feature,label)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x


class Mish_func(torch.autograd.Function):
    """from: https://github.com/tyunist/memory_efficient_mish_swish/blob/master/mish.py"""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]

        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)


def replace_activations(model, existing_layer, new_layer):
    """A function for replacing existing activation layers"""

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model


def get_nfnet_image_embeddings(image_paths, n_classes, model_name='eca_nfnet_l0'):
    embeds = []

    model = ShopeeModelNfnet(model_name=model_name, n_classes=n_classes)
    model.eval()

    if model_name == 'eca_nfnet_l0':
        model = replace_activations(model, torch.nn.SiLU, Mish())

    print(f'nfnet_weight_path={CFG.nfnet_weight_path}')
    model.load_state_dict(torch.load(CFG.nfnet_weight_path))
    device = torch.device("cuda")
    model = model.to(device)

    image_dataset = ShopeeNfnetDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=12,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    with torch.no_grad():
        for img, label in image_loader:
            img = img.cuda()
            label = label.cuda()
            feat = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    del embeds
    gc.collect()
    return image_embeddings


def most_common_preds(x, threshold):
    l = np.concatenate([x[CFG.image_preds_loose_column], x[CFG.tfidf_preds_loose_column], x[CFG.sbert_preds_loose_column], x[CFG.b4_image_preds_loose_column]]).tolist()
    c = Counter(l)
    ret = []
    for mc in c.most_common():
        if mc[1] >= threshold:
            ret.append(mc[0])
    return np.unique(np.concatenate([x['preds'], ret]))


def most_common_preds_if_single_preds(x, threshold):
    if len(x['preds']) == 1:
        return most_common_preds(x, threshold)
    else:
        return x['preds']


def run():
    tf_mem_limit(limit=2)
    df = pd.read_csv(CFG.test_file_path)
    if len(df) > 3:
        CFG.GET_CV = False
    del df

    df, image_paths = read_dataset()
    if CFG.holdout:
        n_classes = 8811
    else:
        n_classes = 11014

    image_embeddings = get_single_image_embeddings(image_paths, n_classes, model_name='EfficientNetB3', effnet_weight_path=CFG.effnetb3_weight_path)
    b4_image_embeddings = get_single_image_embeddings(image_paths, n_classes, model_name='EfficientNetB4', effnet_weight_path=CFG.effnetb4_weight_path)
    tfidf_embeddings = get_tfidf_embeddings(df)
    sbert_embeddings = get_sbert_embeddings(df)

    # effnet B3
    df[CFG.image_preds_strict_column], df[CFG.image_preds_loose_column] = get_neighbors(df, image_embeddings, threshold_strict=CFG.image_threshold_strict)
    if CFG.GET_CV:
        np.save('image_embeddings', image_embeddings)
        f1, precision, recall = f1_precision_recall(df['target'], df[CFG.image_preds_strict_column])
        print(f"{CFG.image_preds_strict_column} f1={np.mean(f1)} precision={np.mean(precision)} recall={np.mean(recall)}")

    # effnet B4
    df[CFG.b4_image_preds_strict_column], df[CFG.b4_image_preds_loose_column] = get_neighbors(df, b4_image_embeddings, threshold_strict=CFG.b4_image_threshold_strict)
    if CFG.GET_CV:
        np.save('b4_image_embeddings', b4_image_embeddings)
        f1, precision, recall = f1_precision_recall(df['target'], df[CFG.b4_image_preds_strict_column])
        print(f"{CFG.b4_image_preds_strict_column} f1={np.mean(f1)} precision={np.mean(precision)} recall={np.mean(recall)}")

    # tfidf
    df[CFG.tfidf_preds_strict_column], df[CFG.tfidf_preds_loose_column] = get_text_preds(df, tfidf_embeddings, threshold_strict=CFG.tfidf_threshold_strict)
    if CFG.GET_CV:
        np.save('tfidf_embeddings', tfidf_embeddings)
        f1, precision, recall = f1_precision_recall(df['target'], df[CFG.tfidf_preds_strict_column])
        print(f"{CFG.tfidf_preds_strict_column} f1={np.mean(f1)} precision={np.mean(precision)} recall={np.mean(recall)}")

    # sbert
    df[CFG.sbert_preds_strict_column], df[CFG.sbert_preds_loose_column] = get_text_preds(df, cupy.asarray(sbert_embeddings), threshold_strict=CFG.sbert_threshold_strict)
    if CFG.GET_CV:
        np.save('sbert_embeddings', sbert_embeddings)
        f1, precision, recall = f1_precision_recall(df['target'], df[CFG.sbert_preds_strict_column])
        print(f"{CFG.sbert_preds_strict_column} f1={np.mean(f1)} precision={np.mean(precision)} recall={np.mean(recall)}")

    # strict concat
    df['preds'] = df.apply(lambda x: np.unique(np.concatenate([x[CFG.image_preds_strict_column], x[CFG.tfidf_preds_strict_column], x[CFG.sbert_preds_strict_column], x[CFG.b4_image_preds_strict_column]])), axis=1)

    # loose intersection
    df['preds'] = df.apply(lambda x: most_common_preds(x, 3), axis=1)
    df['preds'] = df.apply(lambda x: most_common_preds_if_single_preds(x, 2), axis=1)

    # loose union
    df['preds_loose_union'] = df.apply(lambda x: np.unique(np.concatenate([x[CFG.image_preds_loose_column], x[CFG.tfidf_preds_loose_column], x[CFG.sbert_preds_loose_column], x[CFG.b4_image_preds_loose_column]])), axis=1)
    df['preds'] = df.apply(lambda x: replace_single_preds(x), axis=1)

    df['matches'] = df['preds'].apply(lambda x: ' '.join(x))
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    if CFG.GET_CV:
        f1, precision, recall = f1_precision_recall(df['target'], df['preds'])
        print(f"final f1={np.mean(f1)} precision={np.mean(precision)} recall={np.mean(recall)}")
        array_columns = []
        tmp = df.iloc[0]
        for c in df.columns.values:
            if isinstance(tmp[c], np.ndarray) or isinstance(tmp[c], list):
                array_columns.append(c)
        for column in array_columns:
            df[column] = df[column].apply(lambda x: ' '.join(x))
        df.to_csv('processed_train.csv')
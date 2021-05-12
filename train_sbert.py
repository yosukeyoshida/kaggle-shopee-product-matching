import math
from tqdm.notebook import tqdm
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
try:
    from util import preprocess_title
except ImportError:
    pass


class CFG:
    NUM_WORKERS = 4
    TRAIN_BATCH_SIZE = 32
    EPOCHS = [25]  # 25
    SEED = 2020
    LR = 5e-5
    loss_module = 'arcface'
    input_dir = 'drive/MyDrive/Colab Notebooks/shopee-product-matching/data/input'
    train_file_path = os.path.join(input_dir, 'shopee-product-matching', 'train.csv')
    train_holdout_file_path = os.path.join(input_dir, 'shopee-product-matching', 'train_holdout.csv')
    train_images_dir = os.path.join(input_dir, 'shopee-product-matching', 'train_images')
    output_dir = 'drive/MyDrive/Colab Notebooks/shopee-product-matching/data/output/sbert/20210421_holdout'
    transformer_model = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(transformer_model)
    holdout = True


def makedir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fetch_loss():
    loss = nn.CrossEntropyLoss()
    return loss


class ShopeeDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv.reset_index()

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        text = row.title
        text = CFG.TOKENIZER(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        return input_ids, attention_mask, torch.tensor(row.label_group)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

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
        output *= self.s

        return output


class ShopeeNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='bert-base-uncased',
                 pooling='mean_pooling',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
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

        self.pooling = pooling
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.relu = nn.ReLU()
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes, s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, input_ids, attention_mask, label):
        feature = self.extract_feat(input_ids, attention_mask)
        if self.loss_module == 'arcface':
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        features = x[0]
        features = features[:, 0, :]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)
            features = self.relu(features)

        return features


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:

        batch_size = d[0].shape[0]

        input_ids = d[0]
        attention_mask = d[1]
        targets = d[2]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask, targets)

        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            scheduler.step()

    return loss_score


def run():
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
    data['filepath'] = data['image'].apply(lambda x: os.path.join(CFG.train_images_dir, x))
    encoder = LabelEncoder()
    data['label_group'] = encoder.fit_transform(data['label_group'])

    device = torch.device("cuda")

    for epochs in CFG.EPOCHS:
        print(f'epochs={epochs} start')
        train_dataset = ShopeeDataset(csv=data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=CFG.TRAIN_BATCH_SIZE,
            pin_memory=True,
            drop_last=True,
            num_workers=CFG.NUM_WORKERS
        )

        model_params = {
            'n_classes': n_classes,
            'model_name': CFG.transformer_model,
            'pooling': 'clf',
            'use_fc': False,
            'fc_dim': 512,
            'dropout': 0.0,
            'loss_module': CFG.loss_module,
            's': 30.0,
            'margin': 0.50,
            'ls_eps': 0.0,
            'theta_zero': 0.785
        }
        model = ShopeeNet(**model_params)
        model.to(device)
        criterion = fetch_loss()
        criterion.to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_parameters, lr=CFG.LR)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader) * 2,
            num_training_steps=len(train_loader) * epochs
        )
        best_loss = 10000
        for epoch in range(epochs):
            train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler=scheduler, epoch=epoch)

            if train_loss.avg < best_loss:
                best_loss = train_loss.avg
                if CFG.holdout:
                    output_file_name = f'sentence_transfomer_xlm_best_loss_num_epochs_{epochs}_{CFG.loss_module}_holdout.bin'
                else:
                    output_file_name = f'sentence_transfomer_xlm_best_loss_num_epochs_{epochs}_{CFG.loss_module}.bin'
                torch.save(model.state_dict(), os.path.join(CFG.output_dir, output_file_name))

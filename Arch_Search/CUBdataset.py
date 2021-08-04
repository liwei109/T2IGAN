import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torchvision import transforms
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from nltk.tokenize import RegexpTokenizer
# from pytorch_pretrained_bert import BertTokenizer
from PIL import Image

import cfg

args = cfg.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data  # data是一次数据迭代 一张图一个caption，data为10个数据

    # sort data by the length in a decreasing order # 文本长度递减顺序排序
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if args.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if args.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(args.branch_num):
        # print(imsize[i])
        if i < (args.branch_num - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret  # 64 128 256 three types images


class TextDataset(Dataset):
    """
    Text Dataset
    Based on:
        https://github.com/taoxugit/AttnGAN/blob/master/code/datasets.py
    """
    tokenizer = RegexpTokenizer(r'\w+')

    def __init__(self, data_dir, split='train',
                 base_size=8,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = args.captions_per_image  # 10

        self.imsize = []
        for i in range(args.branch_num):
            self.imsize.append(base_size)  # 64 128 256
            base_size = base_size * 2      # base = 8 16 32 64

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()  # 获取每张图片所对应box，如：'058.Pigeon_Guillemot/Pigeon_Guillemot_0089_40008': [77, 27, 328, 305]
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
        self.wordtoix, self.n_words = self.load_text_data(data_dir, split)  # get captions

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011', 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011', 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokens = self.tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))

        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        filepath = os.path.join(data_dir, 'captions.pickle')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)  # 获取所有caption

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            # encoding = latin1 because of incompatability issue b/w python 2 and python 3
            class_id = pickle.load(open(data_dir + '/class_info.pickle', 'rb'), encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((args.words_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= args.words_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:args.words_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = args.words_num
        return x, x_len

    def get_imgs(self, img_path, imsize, bbox=None,
                 transform=None, normalize=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if transform is not None:
            img = transform(img)

        ret = []
        for i in range(args.branch_num):
            # print(imsize[i])
            # if i < (args.branch_num - 1):
            re_img = transforms.Scale(imsize[i])(img)
            # else:
            #     re_img = img
            ret.append(normalize(re_img))

        return ret

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = self.get_imgs(img_name, self.imsize,
                             bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = np.random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key

    def __len__(self):
        return len(self.filenames)


class CubDataset(object):
    def __init__(self, args, split_dir="train", batchsize=10):
        imsize = 256
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        self.dataset = TextDataset(args.data_dir, split_dir,
                                   base_size=args.base_imgsize,
                                   transform=image_transform)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batchsize,
            drop_last=True, shuffle=True, num_workers=4)
        
        


if __name__ == "__main__":
    train_data = CubDataset(args, split_dir='train', batchsize=args.train_batchsize)
    train_loader = train_data.dataloader

    # test_data = CubDataset(args, split_dir='test', batchsize=args.test_batchsize)
    # test_loader = test_data.dataloader

    # print("len(train_loader): ", len(train_loader))
    # print("len(test_loader): ", len(test_loader))

    # 导入text_encoder编码器
    from text_encoder import RNN_ENCODER

    text_encoder = RNN_ENCODER(ntoken=train_data.dataset.n_words, nhidden=args.embedding_len)
    state_dict = torch.load(args.Net_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', args.Net_E)
    text_encoder.eval()
    if args.CUDA:
        text_encoder = text_encoder.cuda()

    # 数据迭代
    train_iter = iter(train_loader)
    step = 0

    while step < len(train_loader):  # 885 origin: 8855
        data = train_iter.next()
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        # captions encode
        hidden = text_encoder.init_hidden(bsz=args.train_batchsize)  # batchsize
        words_embs, sent_embs = text_encoder(captions, cap_lens, hidden)

        step += 1
        print("train step: ", step)
    
    print("###########end train load##########")

    # test_iter = iter(test_loader)
    # step = 0

    # while step < len(test_loader):  # 885 origin: 8855
    #     data = test_iter.next()
    #     imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

    #     # captions encode
    #     hidden = text_encoder.init_hidden(bsz=args.test_batchsize)  # batchsize
    #     words_embs, sent_embs = text_encoder(captions, cap_lens, hidden)

    #     step += 1
    #     print("test step: ", step)


    # print("###########end test load ##########")
    # print("###########start again train load ##########")

    train_data1 = CubDataset(args, split_dir='train', batchsize=args.train_batchsize)
    train_loader1 = train_data.dataloader
    train_iter1 = iter(train_loader1)
    step = 0

    while step < len(train_loader1):  # 885 origin: 8855
        data = train_iter1.next()
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

        # captions encode
        hidden = text_encoder.init_hidden(bsz=args.train_batchsize)  # batchsize
        words_embs, sent_embs = text_encoder(captions, cap_lens, hidden)

        step += 1
        print("train step: ", step)
    # while 1:
    print("########### End ##########")


# class TextBertDataset(TextDataset):
#     """
#     Text dataset on Bert
#     https://github.com/huggingface/pytorch-pretrained-BERT
#     """
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)        # Load pre-trained model tokenizer (vocabulary)

#     def load_captions(self, data_dir, filenames):
#         all_captions = []
#         for i in range(len(filenames)):
#             cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
#             with open(cap_path, "r") as f:
#                 captions = f.read().split('\n')
#                 cnt = 0
#                 for cap in captions:
#                     if len(cap) == 0:
#                         continue
#                     # picks out sequences of alphanumeric characters as tokens
#                     # and drops everything else
#                     tokens = self.tokenizer.tokenize(cap.lower())
#                     # print('tokens', tokens)
#                     if len(tokens) == 0:
#                         print('cap', cap)
#                         continue

#                     tokens_new = []
#                     for t in tokens:
#                         t = t.encode('ascii', 'ignore').decode('ascii')
#                         if len(t) > 0:
#                             tokens_new.append(t)
#                     all_captions.append(tokens_new)
#                     cnt += 1
#                     if cnt == self.embeddings_num:
#                         break
#                 if cnt < self.embeddings_num:
#                     print('ERROR: the captions for %s less than %d'
#                           % (filenames[i], cnt))

#         return all_captions

#     def load_text_data(self, data_dir, split):
#         train_names = self.load_filenames(data_dir, 'train')
#         test_names = self.load_filenames(data_dir, 'test')
#         filepath = os.path.join(data_dir, 'bert_captions.pickle')
#         if not os.path.isfile(filepath):
#             train_captions = self.load_captions(data_dir, train_names)
#             test_captions = self.load_captions(data_dir, test_names)

#             train_captions, test_captions, ixtoword, wordtoix, n_words = \
#                 self.build_dictionary(train_captions, test_captions)
#             with open(filepath, 'wb') as f:
#                 pickle.dump([train_captions, test_captions,
#                              ixtoword, wordtoix], f, protocol=2)
#                 print('Save to: ', filepath)
#         else:
#             with open(filepath, 'rb') as f:
#                 x = pickle.load(f)
#                 train_captions, test_captions = x[0], x[1]
#                 ixtoword, wordtoix = x[2], x[3]
#                 del x
#                 n_words = len(ixtoword)
#                 print('Load from: ', filepath)
#         if split == 'train':
#             # a list of list: each list contains
#             # the indices of words in a sentence
#             captions = train_captions
#             filenames = train_names
#         else:  # split=='test'
#             captions = test_captions
#             filenames = test_names
#         return filenames, captions, ixtoword, wordtoix, n_words


#     def build_dictionary(self, train_captions, test_captions):
#         """
#         Tokenize according to bert model
#         """
#         captions = train_captions + test_captions
#         ixtoword = {}
#         wordtoix = {}


#         train_captions_new = []
#         for sent in train_captions:
#             indexed_tokens = self.tokenizer.convert_tokens_to_ids(sent)
#             train_captions_new.append(indexed_tokens)
#             for idx, word in zip(indexed_tokens, sent):
#                 wordtoix[word] = idx
#                 ixtoword[idx] = word

#         test_captions_new = []
#         for sent in test_captions:
#             indexed_tokens = self.tokenizer.convert_tokens_to_ids(sent)
#             test_captions_new.append(indexed_tokens)
#             for idx, word in zip(indexed_tokens, sent):
#                 wordtoix[word] = idx
#                 ixtoword[idx] = word

#         return [train_captions_new, test_captions_new,
#                 ixtoword, wordtoix, len(ixtoword)]

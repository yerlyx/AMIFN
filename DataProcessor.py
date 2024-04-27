import torch.utils.data as Data
from torchvision import transforms
from tqdm import tqdm
import pickle
import os
from PIL import Image

import spacy
import numpy as np
# 加载Spacy模型
nlp = spacy.load('en_core_web_sm')
# def get_adjacency_matrix(text):
#     # 对文本进行依赖关系分析
#     doc = nlp(text)
#     # 获取文本中的词语数量
#     num_tokens = len(doc)
#     # 创建空的邻接矩阵
#     adjacency_matrix = np.zeros((num_tokens, num_tokens))
#     # 填充邻接矩阵
#     for token in doc:
#         # 获取当前词语的索引和依赖关系的头部索引
#         token_index = token.i
#         head_index = token.head.i
#         # 设置邻接矩阵中对应位置为1，表示存在依赖关系
#         adjacency_matrix[token_index][head_index] = 1
#     return adjacency_matrix

def image_process(image_path, itransform):
    image = Image.open(image_path).convert('RGB')
    image = itransform(image)
    return image

class MyDataset(Data.Dataset):
    def __init__(self,data_dir,imagefeat_dir,tokenizer,max_seq_len,img_feat_dim=2048,crop_size=224):
        self.imagefeat_dir=imagefeat_dir
        self.tokenizer=tokenizer
        self.sentiment_label_list=self.get_sentiment_labels()
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.number = len(self.examples)
        self.img_feat_dim = img_feat_dim
        self.crop_size = crop_size
        self.data_dir = data_dir

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line,index)   

    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for key,value in tqdm(dict.items(),desc="CreatExample"):
            examples.append(value)
        return examples

    def get_sentiment_labels(self):
        return ["0","1","2"]


    def transform(self,line,index):
        max_seq_len =self.max_seq_len
        value=line
        text_a = value['sentence'] 
        text_b = value['aspect']
        graph_id = index
        filename = self.data_dir.rstrip('.pkl')
        fin = open(filename + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        adj = np.pad(idx2graph[graph_id], \
                                  ((0, max_seq_len - idx2graph[graph_id].shape[0]),
                                   (0, max_seq_len - idx2graph[graph_id].shape[0])), 'constant')
        target_ids = self.tokenizer(text_b.lower())['input_ids']  # <s>text_b</s>
        target_mask = [1] * len(target_ids)
        input_ids=self.tokenizer(text_a.lower(),text_b.lower())['input_ids']   #  <s>text_a</s></s>text_b</s>
        input_mask=[1]*len(input_ids)
        padding_id = [1]*(max_seq_len-len(input_ids)) #<pad> :1
        padding_mask=[0]*(max_seq_len-len(input_ids))
        input_ids += padding_id
        input_mask += padding_mask
        tokens=self.tokenizer.decode(input_ids)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len


        padding_idt = [1] * (max_seq_len - len(target_ids))  # <pad> :1
        padding_maskt = [0] * (max_seq_len - len(target_ids))
        target_ids += padding_idt
        target_mask += padding_maskt
        tokenst = self.tokenizer.decode(target_ids)
        assert len(target_ids) == max_seq_len
        assert len(target_mask) == max_seq_len

        img_id = value['iid']
        img_feat = read_pic(self.imagefeat_dir,img_id,self.crop_size)
        sentiment_label=-1
        sentiment_label_map = {label: i for i, label in enumerate(self.sentiment_label_list)}
        senti=value['sentiment']
        if senti:
            sentiment_label=sentiment_label_map[senti]

        return tokens,tokenst,input_ids,input_mask, target_ids,target_mask, sentiment_label,img_id,img_feat,adj



def read_pic(imagefeat_dir,img_id,crop_size):
    itransform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if 'twitter' in imagefeat_dir.lower():
        img_id2 = img_id + '.jpg'
        image_path = os.path.join(imagefeat_dir, img_id2)

    if not os.path.exists(image_path):
        print(image_path)
    try:
        img_feat = image_process(image_path, itransform)
    except:
        # count += 1
        # print('image has problem!')
        image_path_fail = os.path.join(imagefeat_dir, '17_06_4705.jpg')
        img_feat = image_process(image_path_fail, itransform)
    return img_feat




def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    count=0
    fout = open(filename.rstrip('.txt') + '.graph', 'wb')
    for i in range(0, len(lines), 4):
        sentence = lines[i].lower().strip()
        if len(sentence) > 150:
            sentence = sentence[:150]
        adj_matrix = dependency_adj_matrix(sentence)
        idx2graph[count] = adj_matrix
        count += 1
    pickle.dump(idx2graph, fout)
    print('done !!!!' + filename)
    fout.close()

def convert_to_pkl(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    data = {}
    fout = open(filename.rstrip('.txt') + '.pkl', 'wb')
    count = 0
    for i in range(0, len(lines), 4):
        sentence = lines[i].lower().strip()
        aspect = lines[i + 1].lower().strip()
        sentiment = str(int(lines[i + 2].strip())+1)
        iid = lines[i + 3].strip().split('.')[0]
        if len(sentence) > 150:
            sentence = sentence[:150]
        my_dict={
            'iid': iid,
            'sentence': sentence,
            'aspect': aspect,
            'sentiment': sentiment,
        }
        data[count] = my_dict
        count += 1
    pickle.dump(data, fout)
    print('---done !!!' + filename)
    fout.close()
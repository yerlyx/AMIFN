from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm
import torch.nn.functional as F
from transformers import RobertaModel, AutoConfig
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class AMIFN(nn.Module):
    def __init__(self, args,img_feat_dim=2048):
        super().__init__()
        self.args = args
        self.img_feat_dim = img_feat_dim
        config = AutoConfig.from_pretrained(args.roberta_model_dir)
        self.hidden_dim = config.hidden_size
        self.roberta = RobertaModel.from_pretrained(args.roberta_model_dir)
        self.target_roberta = RobertaModel.from_pretrained(args.roberta_model_dir)
        self.feat_linear = nn.Linear(self.img_feat_dim, self.hidden_dim)
        self.img_self_attn = BertSelfEncoder(config, layer_num=1)
        self.v2t = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.dropout1 = nn.Dropout(0.3)
        self.gather = nn.Linear(self.hidden_dim, 1)
        self.dropout2 = nn.Dropout(0.3)
        self.pred = nn.Linear(49, 2)
        self.pred2=nn.Linear(128,2)
        self.ce_loss = nn.CrossEntropyLoss()
        self.t2v = BertCrossEncoder_AttnMap(config, layer_num=1)
        #BertCrossEncoder_AttnMap返回两个值 return all_encoder_layers,all_attn_maps
        ##  基于方面的文本
        self.ta2t = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.ta2tv_gcn = BertCrossEncoder_AttnMap(config, layer_num=1)
        self.senti_selfattn = BertSelfEncoder(config, layer_num=1)


        self.first_pooler = BertPooler(config)  # BertPooler 取hidden_states 第一个词 [batch_size,hidden_size];Linear(hidden_size,hidden_size);Tanh()激活
        self.senti_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.senti_detc  = nn.Linear(self.hidden_dim, 3)
        self.senti_detc2 = nn.Linear(self.hidden_dim * 3, 3)
        self.gc1 = GraphConvolution(768, 768)
        self.gc2 = GraphConvolution(768, 768)
        self.cls_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.add49linear=nn.Linear(args.max_seq_length+49,args.max_seq_length)
        self.linear49seq = nn.Linear(49, args.max_seq_length)
        self.s2v=BertCrossEncoder_AttnMap(config, layer_num=1)
        self.init_weight()

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name):  # linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name):
                module.bias.data.zero_()

    def forward(self, img_id,
                input_ids, input_mask, target_ids,target_mask,img_feat,adj):
        # input_ids,input_mask : [N, L]
        #             img_feat : [N, 49, 2048]


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, seq = input_ids.size()
        # text feature
        roberta_output = self.roberta(input_ids, input_mask)
        sentence_output = roberta_output.last_hidden_state

        # aspect feature
        target_roberta_output = self.target_roberta(target_ids, target_mask)
        target_output = target_roberta_output.last_hidden_state

        img_feat_real = img_feat.view(-1, 2048, 49).permute(0, 2, 1)
        img_feat_ = self.feat_linear(img_feat_real)  # [N, 49, 2048] ->[N, 49, 768]
        image_mask = torch.ones((batch_size, 49)).to(device)
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0

        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0

        extended_target_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_target_mask = extended_target_mask.to(dtype=next(self.parameters()).dtype)
        extended_target_mask = (1.0 - extended_target_mask) * -10000.0

        target_aware_sentence, _ = self.ta2t(sentence_output,
                                            target_output,
                                            extended_target_mask,
                                            output_all_encoded_layers=False)
        target_aware_sentence = target_aware_sentence[-1]  # [N,l,768]

        target_aware_image, _ = self.t2v(target_aware_sentence,
                                         img_feat_,
                                         extended_image_mask,
                                         output_all_encoded_layers=False)  # image query sentence
        target_aware_image = target_aware_image[-1]  # [N,laspect,768]

        hs_hi_mixed_feature = torch.cat((sentence_output, img_feat_), dim=1)
        hs_hi_mask = torch.cat((input_mask, image_mask), dim=-1).to(device)
        extended_hs_hi_mask = hs_hi_mask.unsqueeze(1).unsqueeze(2)
        extended_hs_hi_mask = extended_hs_hi_mask.to(dtype=next(self.parameters()).dtype)
        extended_hs_hi_mask = (1.0 - extended_hs_hi_mask) * -10000.0
        hs_hi_mixed_output = self.senti_selfattn(hs_hi_mixed_feature, extended_hs_hi_mask)  # [N, L+49, 768]
        hs_hi_mixed_output = hs_hi_mixed_output[-1]
        hs_hi_mixed_output = hs_hi_mixed_output.permute(0, 2, 1)  # [n,768,L+49]
        hs_hi_mixed_output = self.add49linear(hs_hi_mixed_output)  # [n,768,L]
        hs_hi_mixed_output = hs_hi_mixed_output.permute(0, 2, 1)  # [n,L,768]

        gathered_target_aware_image = self.gather(self.dropout1(
            target_aware_image)).squeeze(2)  # [N,la,768]->[N,la,1] ->[N,la]
        rel_pred = self.pred2(self.dropout2(
            gathered_target_aware_image))  # [N,2]

        gate = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).\
            expand(batch_size,self.args.max_seq_length).unsqueeze(2).\
            expand(batch_size,self.args.max_seq_length,self.hidden_dim)
        if self.args.addgate == 1:
            gated_target_aware_image = gate * target_aware_image  # [N,l,768]
        else:
            gated_target_aware_image = target_aware_image  # [N,l,768]
        if self.args.addGCN == 1:
            hs_hi_mixed_output = F.relu(self.gc1(hs_hi_mixed_output, adj))
            target_aware_image_gcn = F.relu(self.gc2(hs_hi_mixed_output, adj))
        else:
            target_aware_image_gcn = hs_hi_mixed_output
        target_aware_image_gcn_asi, _ = self.ta2tv_gcn(target_aware_sentence,
                                                       target_aware_image_gcn,
                                                       extended_target_mask)
        target_aware_image_gcn_asi = target_aware_image_gcn_asi[-1]  #[N,l,768]

        asi_mean_pooled_output = self.cls_linear(torch.mean(target_aware_image_gcn_asi, dim=1))
        ai_gated_pooled_output = self.cls_linear(torch.mean(gated_target_aware_image, dim=1))
        as_mean_pooled_output = self.cls_linear(torch.mean(target_aware_sentence, dim=1))
        senti_mixed_feature = torch.cat((ai_gated_pooled_output, as_mean_pooled_output), dim=-1)
        senti_mixed_feature = torch.cat((senti_mixed_feature, asi_mean_pooled_output), dim=-1)
        senti_pooled_output = self.senti_dropout(senti_mixed_feature)
        senti_pred = self.senti_detc2(senti_pooled_output)
        return senti_pred




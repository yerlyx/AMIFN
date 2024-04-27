import torch
import logging
import argparse
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import precision_recall_fscore_support
from DataProcessor import *
from model import AMIFN
import resnet.resnet.resnet as resnet
from resnet.resnet.resnet_utils import myResnet

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens,tokenst,input_ids,input_mask, target_ids,target_mask,sentiment_label,\
                        img_id,img_feat,adj=batch

    input_ids=list(map(list, zip(*input_ids)))
    input_mask=list(map(list, zip(*input_mask)))
    target_ids= list(map(list, zip(*target_ids)))
    target_mask = list(map(list, zip(*target_mask)))

    input_ids=torch.tensor(input_ids,dtype=torch.long).to(device)
    input_mask=torch.tensor(input_mask,dtype=torch.long).to(device)
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)
    target_mask = torch.tensor(target_mask, dtype=torch.long).to(device)
    sentiment_label=sentiment_label.to(device).long()
    img_feat=img_feat.to(device).float()
    adj = adj.clone().detach().to(device).float()

    return tokens,tokenst,input_ids,input_mask,target_ids,target_mask,sentiment_label,\
                        img_id,img_feat,adj



def inference_sentiment(model, encoder,test_dataloader, output_dir, logger):
    model.eval()
    encoder.eval()
    nb_eval_examples = 0
    test_senti_acc=0
    senti_true_label_list = []
    senti_pred_label_list = []
    img_id_list=[]
    
    for batch in tqdm(test_dataloader, desc="Testing_SA"):
        tokens, tokenst, input_ids, input_mask, target_ids, target_mask, sentiment_label, \
        img_id, img_feat, adj = post_dataloader(batch)
        with torch.no_grad():
            imgs_f, img_mean, img_att = encoder(img_feat)
            senti_pred = model(
                img_id=img_id,
                input_ids=input_ids,
                input_mask=input_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                img_feat=img_att,
                adj=adj
            )
            
        sentiment_label=sentiment_label.cpu().numpy()
        senti_pred=senti_pred.cpu().numpy()
        senti_true_label_list.append(sentiment_label)
        senti_pred_label_list.append(senti_pred)
        img_id_list.append(img_id)
        tmp_senti_accuracy = accuracy(senti_pred, sentiment_label)
        test_senti_acc += tmp_senti_accuracy

        current_batch_size=input_ids.size()[0]
        nb_eval_examples += current_batch_size


    test_senti_acc = test_senti_acc / nb_eval_examples
    senti_true_label = np.concatenate(senti_true_label_list)
    senti_pred_outputs = np.concatenate(senti_pred_label_list)
    test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

    result = {
            'Test_senti_acc':test_senti_acc,
            'Test_senti_precision':test_senti_precision,
            'Test_senti_recall':test_senti_recall,
            'Test_senti_F_score':test_senti_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='twitter2017',
                        type=str,
                        choices=['twitter2015', 'twitter2017'],
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default= './data/Sentiment_Analysis/ ',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")

    parser.add_argument("--imagefeat_dir",
                        default = '/data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--output_dir",
                        default="./logXMrealTest/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_file",
                        default="/log/pytorch_model.bin",
                        type=str,
                        required=True,
                        help="The input directory where the model has been written.")
    parser.add_argument("--encoder_file",
                        default="/log/pytorch_model.bin",
                        type=str,
                        required=True,
                        help="The input directory where the model has been written.")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--roberta_model_dir", type=str, default="./roberta-base",
                        help='Path to pre-trained RoBerta model.')
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--addgate",
                        default=1,
                        type=int,
                        help="Whether to add gate,when 1 add.")
    parser.add_argument("--addGCN",
                        default=1,
                        type=int,
                        help="Whether to add GCN,when 1 add.")
    parser.add_argument('--resnet_root',
                        default='./resnet/resnet',
                        help='path the pre-trained cnn models')
    parser.add_argument('--fine_tune_cnn', action='store_true',
                        help='fine tune pre-trained CNN if True')
    args = parser.parse_args()
    
  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    args.data_dir=args.data_dir+str(args.dataset).lower()+ '/%s.pkl'
    # args.imagefeat_dir=args.imagefeat_dir+str(args.dataset).lower()
    args.output_dir=args.output_dir+str(args.dataset)+"/"
   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_logger_file=os.path.join(args.output_dir,'log.txt')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename=output_logger_file)
    logger = logging.getLogger(__name__)
    
    logger.info("dataset:{} ".format(args.dataset))
    logger.info("model_dir:{}  ".format(args.model_file))
    logger.info("encoder_dir:{}  ".format(args.encoder_file))
    logger.info(args)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.model_file)
    model = AMIFN(args)
    model.load_state_dict(model_state_dict)
    model.to(device)

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder_state_dict = torch.load(args.encoder_file)
    encoder = myResnet(net, args.fine_tune_cnn, device)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    test_dataset_SA = MyDataset(args.data_dir%str('test'),args.imagefeat_dir,tokenizer,max_seq_len=args.max_seq_length)
    test_dataloader_SA = Data.DataLoader(dataset=test_dataset_SA,shuffle=False, batch_size=args.eval_batch_size,num_workers=0)

    inference_sentiment(model =model,
                        encoder=encoder,
                        test_dataloader = test_dataloader_SA,
                        output_dir =args.output_dir,
                        logger = logger)

if __name__ == "__main__":
    main()

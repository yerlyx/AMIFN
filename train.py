import argparse
import random
import datetime
import torch
import logging
from transformers import RobertaTokenizer
from tqdm import tqdm, trange

from sklearn.metrics import precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
import resnet.resnet.resnet as resnet
from resnet.resnet.resnet_utils import myResnet
from DataProcessor import *
from model import AMIFN
from optimization import BertAdam


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokens, tokenst, input_ids, input_mask, target_ids, target_mask, sentiment_label, \
        img_id, img_feat, adj = batch

    input_ids = list(map(list, zip(*input_ids)))
    input_mask = list(map(list, zip(*input_mask)))
    target_ids = list(map(list, zip(*target_ids)))
    target_mask = list(map(list, zip(*target_mask)))

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)
    target_mask = torch.tensor(target_mask, dtype=torch.long).to(device)
    sentiment_label = sentiment_label.to(device).long()
    img_feat = img_feat.to(device).float()
    adj = adj.clone().detach().to(device).float()
    return tokens, tokenst, input_ids, input_mask, target_ids, target_mask, sentiment_label, \
        img_id, img_feat, adj


def main():
    start_time = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S_')
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--dataset",
                        default='twitter2015',
                        type=str,
                        choices=['twitter2015', 'twitter2017',
                                 'animal', 'buildings', 'food', 'goods', 'human', 'plant', 'scencry'],
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default='./data/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")

    parser.add_argument("--imagefeat_dir",
                        default='./data/twitter_images/',  # default ='./data/twitter_images/',
                        type=str,
                        required=True,
                        )
    parser.add_argument("--output_dir",
                        default="./log/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--roberta_model_dir',
                        default='roberta-base',
                        type=str)
    parser.add_argument("--addgate",
                        default=1,
                        type=int,
                        help="Whether to add gate,when 1 add.")
    parser.add_argument("--addGCN",
                        default=1,
                        type=int,
                        help="Whether to add GCN,when 1 add.")
    parser.add_argument("--save",
                        default=True,
                        action='store_true',
                        help="Whether to save model.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--SA_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%  of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=2020,  # 24  
                        help="random seed for initialization")
    parser.add_argument('--resnet_root',
                        default='./resnet/resnet',
                        help='path the pre-trained cnn models')
    parser.add_argument('--fine_tune_cnn', action='store_true',
                        help='fine tune pre-trained CNN if True')
    #  增加一行关于do-train
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    # ---------------------------------------------------------------------------

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_dir = args.data_dir + str(args.dataset).lower() + '/%s.pkl'
    args.output_dir = args.output_dir + str(args.dataset) + "/"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_logger_file = os.path.join(args.output_dir, 'log.txt')
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=output_logger_file)
    logger = logging.getLogger(__name__)

    logger.info("dataset:{}   num_train_epochs:{}".format(args.dataset, args.num_train_epochs))
    logger.info("SA_learning_rate:{}  warmup_proportion:{}".format(args.SA_learning_rate, args.warmup_proportion))
    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model_dir)

    train_dataset_SA = MyDataset(args.data_dir % str('train'), args.imagefeat_dir, tokenizer,
                                 max_seq_len=args.max_seq_length)
    train_dataloader_SA = Data.DataLoader(dataset=train_dataset_SA, shuffle=True, batch_size=args.train_batch_size,
                                          num_workers=0)

    eval_dataset_SA = MyDataset(args.data_dir % str('dev'), args.imagefeat_dir, tokenizer,
                                max_seq_len=args.max_seq_length)
    eval_dataloader_SA = Data.DataLoader(dataset=eval_dataset_SA, shuffle=False, batch_size=args.eval_batch_size,
                                         num_workers=0)


    test_dataset = MyDataset(args.data_dir % str('test'), args.imagefeat_dir, tokenizer,
                             max_seq_len=args.max_seq_length)
    test_dataloader = Data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.eval_batch_size,
                                      num_workers=0)

    train_number = max(train_dataset_SA.number, 0)
    num_train_steps = int(train_number / args.train_batch_size * args.num_train_epochs)

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    model = AMIFN(args)
    model.to(device)
    encoder.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_SA = BertAdam(optimizer_grouped_parameters,
                            lr=args.SA_learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)

    SA_global_step = 0
    nb_tr_steps = 0
    max_senti_acc = 0.0
    max_test_senti_acc = 0
    best_epoch = -1

    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

        logger.info("************************************************** Epoch: " + str(
            train_idx) + " *************************************************************")
        logger.info("  Num examples = %d", train_number)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        ### train
        model.train()
        encoder.train()
        encoder.zero_grad()
        senti_l = 0
        for step, batch_SA in enumerate(tqdm(train_dataloader_SA, desc="Iteration")):

            #### SA
            SA_tokens, SA_tokenst, SA_input_ids, SA_input_mask, SA_target_ids, SA_target_mask, SA_sentiment_label, \
                SA_img_id, SA_img_feat, SA_adj = post_dataloader(batch_SA)
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(SA_img_feat)
            senti_pred = model(img_id=SA_img_id,
                               input_ids=SA_input_ids,
                               input_mask=SA_input_mask,
                               target_ids=SA_target_ids,
                               target_mask=SA_target_mask,
                               img_feat=img_att,
                               adj=SA_adj
                               )

            senti_loss_fct = CrossEntropyLoss()
            sentiment_loss = senti_loss_fct(senti_pred.view(-1, 3), SA_sentiment_label.view(-1))
            loss_SA = sentiment_loss
            loss_SA.backward()
            senti_l += sentiment_loss.item()
            lr_this_step = args.SA_learning_rate * warmup_linear(SA_global_step / num_train_steps,
                                                                 args.warmup_proportion)
            for param_group in optimizer_SA.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_SA.step()
            optimizer_SA.zero_grad()
            SA_global_step += 1
            nb_tr_steps += 1

        senti_l = senti_l / nb_tr_steps
        logger.info("sentiment_loss:%s", senti_l)

        ### dev 
        model.eval()
        encoder.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", eval_dataset_SA.number)  # len(eval_examples)
        logger.info("  Batch size = %d", args.eval_batch_size)

        SA_nb_eval_examples = 0
        senti_acc = 0
        senti_precision, senti_recall, senti_F_score = 0, 0, 0
        senti_true_label_list = []
        senti_pred_label_list = []

        #### SA
        for batch_SA in tqdm(eval_dataloader_SA, desc="Evaluating_SA"):
            SA_tokens, SA_tokenst, SA_input_ids, SA_input_mask, SA_target_ids, SA_target_mask, SA_sentiment_label, \
                SA_img_id, SA_img_feat, SA_adj = post_dataloader(batch_SA)
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(SA_img_feat)
                SA_senti_pred = model(
                    img_id=SA_img_id,
                    input_ids=SA_input_ids,
                    input_mask=SA_input_mask,
                    target_ids=SA_target_ids,
                    target_mask=SA_target_mask,
                    img_feat=img_att,
                    adj=SA_adj
                )
            SA_sentiment_label = SA_sentiment_label.cpu().numpy()
            SA_senti_pred = SA_senti_pred.cpu().numpy()
            senti_true_label_list.append(SA_sentiment_label)
            senti_pred_label_list.append(SA_senti_pred)
            tmp_senti_accuracy = accuracy(SA_senti_pred, SA_sentiment_label)
            senti_acc += tmp_senti_accuracy
            current_batch_size = SA_input_ids.size()[0]
            SA_nb_eval_examples += current_batch_size
        senti_acc = senti_acc / SA_nb_eval_examples

        senti_true_label = np.concatenate(senti_true_label_list)
        senti_pred_outputs = np.concatenate(senti_pred_label_list)
        senti_precision, senti_recall, senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {
                  'Dev_senti_acc': senti_acc,
                  'Dev_senti_precision': senti_precision,
                  'Dev_senti_recall': senti_recall,
                  'Dev_senti_F_score': senti_F_score,
                  }
        logger.info("***** Dev Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        ### test
        model.eval()
        encoder.eval()
        logger.info("***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", test_dataset.number)
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_test_examples = 0

        test_senti_acc = 0
        test_senti_true_label_list = []
        test_senti_pred_label_list = []
        for batch in tqdm(test_dataloader, desc="Testing"):
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

            sentiment_label = sentiment_label.cpu().numpy()
            senti_pred = senti_pred.cpu().numpy()
            test_senti_true_label_list.append(sentiment_label)
            test_senti_pred_label_list.append(senti_pred)
            tmp_senti_accuracy = accuracy(senti_pred, sentiment_label)
            test_senti_acc += tmp_senti_accuracy

            current_batch_size = input_ids.size()[0]
            nb_test_examples += current_batch_size

        test_senti_acc = test_senti_acc / nb_test_examples

        senti_true_label = np.concatenate(test_senti_true_label_list)
        senti_pred_outputs = np.concatenate(test_senti_pred_label_list)
        test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {
            'Test_senti_acc': test_senti_acc,
            'Test_senti_F_score': test_senti_F_score,
            'Test_senti_precision': test_senti_precision,
            'Test_senti_recall': test_senti_recall
        }
        logger.info("***** Test Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        # save model
        if senti_acc >= max_senti_acc:
            # Save a trained model
            if args.save:
                model_to_save = model.module if hasattr(model, 'module') else model
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
            max_senti_acc = senti_acc
            corresponding_acc = senti_acc
            corresponding_p = senti_precision
            corresponding_r = senti_recall
            corresponding_f = senti_F_score
            best_epoch = train_idx

        if test_senti_acc >= max_test_senti_acc:
            max_test_senti_acc = test_senti_acc
            test_acc = test_senti_acc
            test_p = test_senti_precision
            test_r = test_senti_recall
            test_f = test_senti_F_score
            best_test_epoch = train_idx

    logger.info("max_dev_senti_acc: %s ", max_senti_acc)
    logger.info("corresponding_sentiment_acc: %s ", corresponding_acc)
    logger.info("correspondingt_sentiment_F_score: %s ", corresponding_f)
    logger.info("corresponding_sentiment_precision: %s ", corresponding_p)
    logger.info("corresponding_sentiment_recall: %s ", corresponding_r)
    logger.info("best_dev_epoch: %d", best_epoch)
    # -----------
    logger.info("max_test_senti_acc: %s ", max_test_senti_acc)
    logger.info("test_sentiment_acc: %s ", test_acc)
    logger.info("test_sentiment_F_score: %s ", test_f)
    logger.info("test_sentiment_precision: %s ", test_p)
    logger.info("test_sentiment_recall: %s ", test_r)
    logger.info("best_epoch: %d", best_test_epoch)


if __name__ == "__main__":
    main()

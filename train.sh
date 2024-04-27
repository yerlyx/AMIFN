output_dir="./logXMTest/"
for i in 'twitter2015' #'twitter2017'
do
    echo ${i}
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --imagefeat_dir ../Twitter_image/${i}_images/ \
    --do_train \
    --max_seq_length 128 \
    --output_dir ${output_dir} \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --seed 2020 \
    --roberta_model_dir ./roberta-base
done


for i in  'twitter2015' #'twitter2017'
do
    echo ${i}
    CUDA_VISIBLE_DEVICES=0 python test.py --dataset ${i} \
    --data_dir ./data/Sentiment_Analysis/ \
    --imagefeat_dir ../Twitter_image/${i}_images/ \
    --output_dir ${output_dir} \
    --eval_batch_size 32 \
    --model_file ${output_dir}${i}/pytorch_model.bin\
    --encoder_file ${output_dir}${i}/pytorch_encoder.bin\
    --roberta_model_dir ./roberta-base
done







#
#SI_SqIkv
#for i in  'twitter2017' #'twitter2015'
#do
#    echo ${i}
#    CUDA_VISIBLE_DEVICES=1 python XM_train.py --dataset ${i} \
#    --data_dir ./data/Sentiment_Analysis/ \
#    --imagefeat_dir ./data/twitter_images/ \
#    --do_train \
#    --output_dir ./logs/XMtest${i}/ \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --seed 2020 \
#    --roberta_model_dir ./roberta-base
#done


###----------------XM_train
# 149记得修改回去
#CUDA_VISIBLE_DEVICES=3

# 参数
# AdamW optimizer
# batch size 32
# training epoch 10
# λ1 1  pred_loss_ratio
# λ2 0.5. ranking_loss_ratio
#  TMSC SA_learning_rate 1e-5
# two auxiliary tasks VG_learning_rate 1e-6
# seed 2020

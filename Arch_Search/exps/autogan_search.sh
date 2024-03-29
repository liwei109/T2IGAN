#!/usr/bin/env bash

python -W ignore search.py \
--data_dir '../data/birds' \
--CUDA True \
--branch_num 4 \
--captions_per_image 10 \
--words_num 18 \
--base_imgsize 8 \
--train_batchsize 8 \
--test_batchsize 8 \
--embedding_len 256 \
--RNN_type LSTM \
--Net_E '../DAMSMencoders/bird/text_encoder200.pth' \
--dataset cub \
--bottom_width 4 \
--img_size 256 \
--gen_model shared_gan \
--dis_model shared_gan \
--controller controller \
--latent_dim 200 \
--gf_dim 512 \
--df_dim 64 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.00002 \
--d_lr 0.00002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--ctrl_sample_batch 1 \
--num_candidate 10 \
--topk 5 \
--shared_epoch 15 \
--grow_step1 15 \
--grow_step2 35 \
--grow_step3 60 \
--max_search_iter 90 \
--ctrl_step 30 \
--exp_name autogan_search
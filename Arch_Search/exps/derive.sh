#!/usr/bin/env bash

python train_derived.py \
--data_dir 'data/birds' \
--CUDA True \
--branch_num 3 \
--captions_per_image 10 \
--words_num 18 \
--base_imgsize 64 \
--train_batchsize 5 \
--test_batchsize 5 \
--embedding_len 256 \
--RNN_type LSTM \
--Net_E './DAMSMencoders/bird/text_encoder200.pth' \
-gen_bs 64 \
-dis_bs 64 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 256 \
--max_epoch 100 \
--max_iter 20000 \
--gen_model shared_gan \
--dis_model shared_gan \
--latent_dim 256 \
--gf_dim 128 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--arch 1 1 1 1 1 0 1 1 1 1 0 0 1 2 \
--exp_name retrain_searched
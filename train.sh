torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
main_train.py \
    --world_size 1 \
    --batch_size 10 \
    --data_path "<The location of your combined dataset configuration file>" \
    --epochs 200 \
    --lr 1e-4 \
    --min_lr 3e-7 \
    --weight_decay 0.05 \
    --pretrain_path "<path to uniformer_base_ls_in1k.pth>" \
    --test_data_path "<path to /CASIA1.0>" \
    --warmup_epochs 4 \
    --output_dir ./output_dir/ \
    --log_dir ./output_dir/  \
    --accum_iter 1 \
    --seed 42 \
    --test_period 4 \
    --num_workers 8 \
    2> train_error.log 1>train_log.log

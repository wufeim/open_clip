python -m training.main \
    --train-data /home/data/imagenet1k/train \
    --warmup 1000 \
    --batch-size 2 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 3 \
    --model "dediffusion_ViT-B-32" \
    --report-to "wandb" \
    --coca-contrastive-loss-weight 0 \
    --coca-caption-loss-weight 1 \
    --log-every-n-steps 100

exit

cd open_clip/src
torchrun --nproc_per_node 4 -m training.main \
    --train-data /home/data/imagenet1k/train \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /home/data/imagenet1k/val

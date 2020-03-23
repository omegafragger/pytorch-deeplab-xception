CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.007 --workers 1 --use-sbd --epochs 50 --batch-size 2 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset pascal

cd ..
# train
#python train_net.py \
#        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x.yaml \
#        --resume  \
#        --num-gpus 2

# test
python train_net.py \
        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x.yaml \
        --eval-only \
        --resume \
        --num-gpus 2
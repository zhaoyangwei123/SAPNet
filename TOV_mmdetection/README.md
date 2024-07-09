The SAPNet code is in mmdet/models/detectors/SAPNet.py mmdet/models/roi_heads/SAP_head.py
our GUPs: 8 * RTX3090

# Prerequisites
#install environment following
```sh
conda create -n open-mmlab python=3.9 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
or
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmcv-full ==1.5.0

# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
pip install scikit-image
```


#  Prepare dataset COCO
1. download dataset to data/coco
2. generate point annotation or download point annotation(
[Baidu Yun passwd:6752](https://pan.baidu.com/s/1XF9TneCxByqOJfAaqciP8A?pwd=6752 ) or 
[Google Driver]()],
3. generate SAM_results with point prompt
move annotations/xxx to data/coco/annotations_qc_pt/xxx


#  QC Point generation for coco
## 1.generate QC point annotation
```sh
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export SR=0.25 # size_range
export VERSION=1
export CORNER=""
# export T="val"
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_mask_utils.py "generate_noisept_dataset" \
    "data/coco/annotations/instances_${T}2017.json" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json" \
    --rand_type 'center_gaussian' --range_gaussian_sigma "(${MU[0]},${MU[1]})" --range_gaussian_sigma "(${S[0]},${S[1]})" \
    --size_range "${SR}"
```
## 2.Transfer QC point annotation to 'bbox' and transfer original bbox to 'true_bbox'
### the QC point annotation is transfered to 'bbox' with fixed w and h, which is easy for mmdetection reading and dataset pipeline
### the original bbox is transfered to 'true_bbox', which is the real box ground-turth
```sh
export VERSION=1
export MU=(0 0)
export S=(0.25 0.25)  # sigma
export RS=0.25
export CORNER=""
export WH=(64 64)
export T="train"
PYTHONPATH=. python huicv/coarse_utils/noise_data_utils.py "generate_pseudo_bbox_for_point" \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse.json"  \
    "data/coco/coarse_annotations_new/quasi-center-point-${MU[0]}-${MU[1]}-${S[0]}-${S[1]}-${SR}_${VERSION}/${CORNER}/qc_instances_${T}2017_coarse_with_gt.json"  \
    --pseudo_w ${WH[0]} --pseudo_h ${WH[1]}
```

## 3.For other dataset, we can transform the annotation style to coco json style and use the same way, after this, you can genertae annotation 

# Train, Test and Visualization

## Take COCO as example
### Prepare trained model 
1. move coco dataset (2017 version) or make a soft link to data/coco
2. download weight and annotation(coco and voc2012SBD) from [Baidu Yun(passwd:u5fu)](链接: https://pan.baidu.com/s/1d1sWCpytBlynQ8NUd1O7bw) or [Google Driver](https://drive.google.com/drive/folders/1TKy9DpvNAqr5IdMG85NjDFp_X7iUryk-?usp=drive_link) ,



### Train && inference
```open to the work path: SAPNet/TOV_mmdetection```
1. SAPNet
    ```shell script
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/COCO/SAPNet/SAPNet_r50_fpn_1x_coco_ms.py 8 
    ```

2. inference with trained P2BNet to get pseudo box and train FasterRCNN with pseudo box
    ```shell script
	work_dir='../TOV_mmdetection_cache/work_dir/coco/' && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_test.sh configs2/COCO/P2BNet/SAPNet_r50_fpn_1x_coco.py epoch_12.pth 8 
    ```





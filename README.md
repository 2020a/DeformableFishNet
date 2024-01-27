# DeformableFishNet
 
# 1. Install
    pip install -r requirements.txt
    python setup.py develop
    pip install timm thop efficientnet_pytorch einops -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"

# 2. Train
You can use */train.py* to train the model. The following is the parameter configuration in */train.py*.

    parser.add_argument('--yaml', type=str, default='ultralytics/models/v8/DeformableFishNet.yaml', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='pretrained model path')
    parser.add_argument('--cfg', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/coco128.yaml', help='data yaml path')
    
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--unamp', action='store_true', help='Unuse Automatic Mixed Precision (AMP) training')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', type=str, default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    parser.add_argument('--resume', type=str, default='', help='resume training from last checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'NAdam', 'RAdam', 'AdamW', 'RMSProp', 'auto'], default='SGD', help='optimizer (auto -> ultralytics/yolo/engine/trainer.py in build_optimizer funciton.)')
    parser.add_argument('--close_mosaic', type=int, default=0, help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--info', action="store_true", help='model info verbose')
    
    parser.add_argument('--save', type=str2bool, default='True', help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', action="store_true", default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds during training for loggers')
    
    # Segmentation
    parser.add_argument('--overlap_mask', type=str2bool, default='True', help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')
    
    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

# 3. Val
You can use */val.py* to verify the model. The following is the parameter configuration in */val.py*.

    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='training model path')
    parser.add_argument('--data', type=str, default='ultralytics/datasets/coco128.yaml', help='data yaml path')
    parser.add_argument('--imgsz', type=int, default=640, help='size of input images as integer')
    parser.add_argument('--batch', type=int, default=16, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split to use for validation, i.e. val, test or train')
    parser.add_argument('--project', type=str, default='runs/val', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_json', action="store_true", help='save results to JSON file')
    parser.add_argument('--save_hybrid', action="store_true", help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, default=0.001, help='object confidence threshold for detection (0.001 in val)')
    parser.add_argument('--iou', type=float, default=0.65, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--half', action="store_true", help='use half precision (FP16)')
    parser.add_argument('--dnn', action="store_true", help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', action="store_true", default=True, help='ave plots during train/val')

# 4. Detect
You can use */detect.py* for detection. The following is the parameter configuration in */detect.py*.

    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='training model path')
    parser.add_argument('--source', type=str, default='ultralytics/assets', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'track'], help='predict mode or track mode')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name (project/name)')
    parser.add_argument('--show', action="store_true", help='show results if possible')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_conf', action="store_true", help='save results with confidence scores')
    parser.add_argument('--show_labels', action="store_true", default=True, help='show object labels in plots')
    parser.add_argument('--show_conf', action="store_true", default=True, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=3, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action="store_true", help='visualize model features')
    parser.add_argument('--augment', action="store_true", help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action="store_true", help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--retina_masks', action="store_true", help='use high-resolution segmentation masks')
    parser.add_argument('--boxes', action="store_true", default=True, help='Show boxes in segmentation predictions')
    parser.add_argument('--save', action="store_true", default=True, help='save result')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml'], help='tracker type, [botsort.yaml, bytetrack.yaml]')

# 5. Model file
1. The network structure of **DeformableFishNet** detection model proposed by us can be found in *ultralytics/models/v8/DeformableFishNet.yaml* .
2. The **efficient global coordinate attention module (EGCA)** module we proposed is in *ultralytics/nn/extra_modules/attention.py* line 1007.
3. The **deformable convolution network based on EGCA (EDCN/EC2f)** module we proposed is in *ultralytics/nn/extra_modules/block.py* line 1497 .
4. The network structure of **EC2f-based feature pyramid network (EDBFPN)** module we proposed is in *ultralytics/models/v8/EDBFPN.yaml* .
5. The **efficient multi-scale decoupling head (EMSD Head)** module we proposed is in *ultralytics/nn/extra_modules/head.py*  line 14.

# 6. Dataset
The **freshwater fish dataset** is under *Freshwater fish/* directory. There are 4691 pictures in our **freshwater fish dataset**, all of which were taken by underwater RGB camera in natural freshwater environment.The fish in the picture belong to two species, koi fish and tilapia.

Tilapia and koi fish are two different fish, and there are some obvious differences in their body surface characteristics. The body surface of tilapia is covered with smooth and hard scales, which are small and closely arranged. Their scales don't reflect light as koi fish does. The body of tilapia is usually oval or slightly flat. They are relatively small, especially compared with adult koi fish. Tilapia has many kinds of fins, including dorsal fin, anal fin, caudal fin and pectoral fin. Their fins are relatively short and strong, and they usually don't have a wide and unique shape like koi fish. Koi fish is famous for its colorful colors and unique patterns, while the color of tilapia is relatively simple, usually gray, silver or brown. Koi fish is bigger and rounder than tilapia.

Using **DeformableFishNet** to train freshwater fish dataset needs to store *Freshwater fish/images/* and *Freshwater fish/labels/* directories under *dataset/* directory.
 


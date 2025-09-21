set data_dirpath=D:/data/coco
set result_dirpath=D:/results/yolov7

python main.py --data-dirpath %data_dirpath%/coco --output-dirpath %result_dirpath% ^
--worker 1 --device cpu --batch-size 2 --data coco.yaml --img 1280 1280 --cfg yolov7-w6.yaml ^
--weights ''  --name yolov7-w6 --hyp hyp.scratch.p6.yaml --correct-exif --print-freq 1000

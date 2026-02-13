python infer/modules/train/preprocess.py $1 48000 43 logs/$1 False 3.7
python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 logs/$1 True
python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 logs/$1 v2 True
python generate_training_files.py $1
python infer/modules/train/train.py -e $1 -sr 48k -f0 1 -bs 23 -g 0 -te 1000 -se 50 -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth -l 1 -c 1 -sw 0 -v v2
python generate_training_index.py $1


import sys

traing_dir = sys.argv[1]
if len(sys.argv) > 1:
    print("Generate files from:", traing_dir)
else:
    print("No arguments passed!")


from random import shuffle
import os

exp_dir = f'logs/{traing_dir}'
gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)

feature_dir = "%s/3_feature768" % (exp_dir)
import numpy as np



f0_dir = "%s/2a_f0" % (exp_dir)
f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
names = (
    set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
    & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0_dir)])
    & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
)

opt = []
for name in names:

    opt.append(
        "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
        % (
            gt_wavs_dir.replace("\\", "\\\\"),
            name,
            feature_dir.replace("\\", "\\\\"),
            name,
            f0_dir.replace("\\", "\\\\"),
            name,
            f0nsf_dir.replace("\\", "\\\\"),
            name,
            0,
        )
    )

fea_dim =  768

for _ in range(2):
    opt.append(
        "logs/mute/0_gt_wavs/mute%s.wav|logs/mute/3_feature%s/mute.npy|logs/mute/2a_f0/mute.wav.npy|logs/mute/2b-f0nsf/mute.wav.npy|%s"
        % ('48k', fea_dim,  0)
    )
shuffle(opt)
with open("%s/filelist.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt))



config_path = "v2/48k.json"
config_save_path = os.path.join(exp_dir, "config.json")
cmd = f'cp configs/v2/48k.json {config_save_path}'
os.system(cmd)


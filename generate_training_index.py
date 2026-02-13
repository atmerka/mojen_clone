import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import faiss

import sys

exp_dir1 =  sys.argv[1]
if len(sys.argv) > 1:
    print("Generate files from:", exp_dir1)
else:
    print("No arguments passed!")


version19= 'v2'
outside_index_root = 'assets/indices'
# exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
exp_dir = "logs/%s" % (exp_dir1)
os.makedirs(exp_dir, exist_ok=True)
feature_dir = (
    "%s/3_feature256" % (exp_dir)
    if version19 == "v1"
    else "%s/3_feature768" % (exp_dir)
)

listdir_res = list(os.listdir(feature_dir))

infos = []
npys = []
for name in sorted(listdir_res):
    phone = np.load("%s/%s" % (feature_dir, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]
if big_npy.shape[0] > 2e5:
    infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
    print("\n".join(infos)) 

    big_npy = (
        MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * 64,
            compute_labels=False,
            init="random",
        )
        .fit(big_npy)
        .cluster_centers_
    )


np.save("%s/total_fea.npy" % exp_dir, big_npy)
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
infos.append("%s,%s" % (big_npy.shape, n_ivf))

index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
# index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
infos.append("training")

index_ivf = faiss.extract_index_ivf(index)  #
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index,
    "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
)
infos.append("adding")

batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
faiss.write_index(
    index,
    "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
)


link = os.symlink
link(
    "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
    % (
        outside_index_root,
        exp_dir1,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    ),
)



#!/bin/bash
# Raw training sources (TRAIN splits only), fetched in parallel on the GPU host.
set -uo pipefail
D=/root/raw; mkdir -p $D/vqa $D/coco $D/m2w; cd $D
( cd vqa && for f in v2_Questions_Train_mscoco v2_Annotations_Train_mscoco v2_Complementary_Pairs_Train_mscoco; do
    aria2c -q -x8 -s8 https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/$f.zip && unzip -qo $f.zip; done; echo VQA_OK ) &
( cd coco && aria2c -q -x16 -s16 http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -qo annotations_trainval2014.zip annotations/instances_train2014.json && echo COCO_ANN_OK ) &
( git clone -q --depth 1 https://github.com/wenhuchen/Table-Fact-Checking tabfact && echo TABFACT_OK ) &
( python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
api = HfApi()
info = api.dataset_info("osunlp/Multimodal-Mind2Web")
files = [s.rfilename for s in info.siblings if s.rfilename.startswith("data/train-")]
print("m2w revision", info.sha, len(files), "train shards")
open("/root/raw/m2w/REVISION", "w").write(info.sha + "\n")
with ThreadPoolExecutor(14) as ex:
    list(ex.map(lambda f: hf_hub_download("osunlp/Multimodal-Mind2Web", f, repo_type="dataset",
                                          revision=info.sha, local_dir="/root/raw/m2w"), files))
print("M2W_OK")
PY
) &
wait
ls -la $D/vqa $D/coco/annotations; du -sh $D/*
echo FETCH_DONE

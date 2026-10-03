#!/bin/bash
# One-time setup on the vast host: the exp/image branch, pinned deps, eval data local.
set -euo pipefail
cd /root
[ -d sd ] || git clone -q -b exp/image https://github.com/Vivek0712/strands-decider sd
git -C sd pull -q
pip install -q -e "sd[vision]" "transformers==5.18.0" "peft==0.21.2" pandas pyarrow jinja2 \
    imagehash matplotlib aiohttp pyyaml 2>&1 | grep -v "root user" || true
apt-get -qq update >/dev/null && apt-get -qq install -y aria2 fonts-dejavu-core fonts-liberation unzip >/dev/null || true
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("StrandsAgents/strands-decider-2B-hobson-v19", revision="bb282d786bc251fd4e3068de3ada9ddbb38127cd")
snapshot_download("Qwen/Qwen3.5-2B-Base", revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
for f in ["data/train-00000-of-00003.parquet"]:
    print(hf_hub_download("BaiqiL/NaturalBench", f, repo_type="dataset", revision="ba41a7d564877a9b64c094b08015ca493cc3e54b"))
print(hf_hub_download("lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet", repo_type="dataset", revision="4db1276663dfa5eb8ad16a52d24c31a09e470896"))
PY
python -c "import torch, transformers; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"
echo SETUP_OK

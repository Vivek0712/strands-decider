#!/bin/bash
# Build the image training set on the GPU host (CPU work, overlaps GPU evals).
set -euo pipefail
B=/root/sd/research/image-track/build
O=/root/data; mkdir -p $O/coco
cd $B
W=${W:-16}
python vqa.py --vqa-dir /root/raw/vqa --out $O
python count.py --instances /root/raw/coco/annotations/instances_train2014.json --out $O --prefer-images $O/vqa_images.txt
cat $O/vqa_images.txt $O/count_images.txt | sort -u | sed 's#^coco/##' > $O/coco_needed.txt
echo "coco images needed: $(wc -l < $O/coco_needed.txt)"
sed 's#^#http://images.cocodataset.org/train2014/#' $O/coco_needed.txt > $O/coco_urls.txt
aria2c -q -j 64 -x 1 --auto-file-renaming=false --allow-overwrite=true -d $O/coco -i $O/coco_urls.txt || true
echo "coco images present: $(ls $O/coco | wc -l)"
python charts.py --out $O --n-charts 2600 --workers $W
python docs.py --out $O --n-docs 1500 --workers $W
python tabfact.py --repo /root/raw/tabfact --out $O --n-tables 1300 --workers $W
python m2w.py --parquets /root/raw/m2w/data/train-*.parquet --out $O --per-file 230 --workers $W
echo BUILD_DONE

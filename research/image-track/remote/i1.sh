#!/bin/bash
# I1: v19 at 448 / pixel budgets, evaluation sets + held-out NaturalBench groups 300-599.
set -uo pipefail
cd /root/sd
R=/root/runs; mkdir -p $R
IJB=/root/ijb/ijb_preview.jsonl
ev() { local name=$1; shift; local t0=$(date +%s)
  python evaluation/vision/run.py --out $R/$name --systems strands --device cuda "$@" > $R/$name.log 2>&1
  echo "$name rc=$? wall=$(( $(date +%s) - t0 ))s" | tee -a $R/walls.txt; }
ev i1-v19-448      --nb-groups 300 --pope 600 --ijb-jsonl $IJB --long-side 448
ev i1-v19-px400k   --nb-groups 300 --pope 600 --ijb-jsonl $IJB --long-side 0 --max-pixels 400000
ev cal-v19-448     --nb-start 300 --nb-groups 300 --pope 0 --no-blind --long-side 448
ev cal-v19-px400k  --nb-start 300 --nb-groups 300 --pope 0 --no-blind --long-side 0 --max-pixels 400000
ev i1-v19-px800k   --nb-groups 300 --pope 600 --ijb-jsonl $IJB --long-side 0 --max-pixels 800000 --no-blind
ev cal-v19-px800k  --nb-start 300 --nb-groups 300 --pope 0 --no-blind --long-side 0 --max-pixels 800000
echo I1_DONE | tee -a $R/walls.txt

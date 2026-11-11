#!/bin/bash
# Fast end-to-end check of generator, method, and both evaluators (tiny iteration counts).
set -e
cd /home/user
rm -rf tmp/smoke && mkdir -p tmp/smoke/ckpt
for exp in synthetic_validation_paper synthetic_large_motion; do
  python method/splannequin.py --variant splannequin --scenes data/train/$exp/*.json \
    --out-dir tmp/smoke/ckpt/splannequin --snap-out-dir tmp/smoke/ckpt/snap_variant \
    --iterations 60 --coarse-iterations 10 --reg-start 20 --l1-start 40 --reg-interval 5 --log-every 20
  python method/splannequin.py --variant static --scenes data/train/$exp/*.json \
    --out-dir tmp/smoke/ckpt/static --iterations 30 --log-every 10
done
python eval/train/evaluate.py --checkpoints tmp/smoke/ckpt --out tmp/smoke/scores_train.json --details tmp/smoke/details_train.json
python eval/test/evaluate.py --checkpoints tmp/smoke/ckpt --out tmp/smoke/scores_test.json
python - <<'EOF'
import json
tr = json.load(open("tmp/smoke/scores_train.json"))
te = json.load(open("tmp/smoke/scores_test.json"))
print("train meta", json.dumps(tr["metadata"]))
print("test meta", json.dumps(te["metadata"])[:600])
for exp, d in tr["experiments"].items():
    for slot, m in d["results"].items():
        print("train", exp, slot, {k: round(v, 4) for k, v in m.items()})
assert not tr["metadata"]["failed_methods"], tr["metadata"]["failed_methods"]
assert "4dgaussians_splannequin" in te["experiments"]["synthetic_validation_paper"]["results"]
print("SMOKE OK")
EOF
echo "[smoke] timing probe: 400 iterations on an 8-scene batch"
python method/splannequin.py --variant splannequin --scenes data/test/synthetic_validation_paper/*.json \
  --out-dir tmp/smoke/timing --iterations 400 --coarse-iterations 100 --reg-start 200 --l1-start 300 --log-every 100


#!/usr/bin/env bash
python convert_visa_to_mvtec_format.py \
    -s ../visa \
    -t ../visa_ \

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir ../visa_ \
    --dest-dir   ../LTN_visa/visa-step_k1-seed${seed} \
    --prune-manifest manifest/visa-nlt/step_k1/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/step_k1/seed${seed}/inject_defects.txt
done

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir ../visa_ \
    --dest-dir   ../LTN_visa/visa-step_k4-seed${seed} \
    --prune-manifest manifest/visa-nlt/step_k4/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/step_k4/seed${seed}/inject_defects.txt
done

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir ../visa_ \
    --dest-dir   ../LTN_visa/visa-pareto-seed${seed} \
    --prune-manifest manifest/visa-nlt/pareto/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/pareto/seed${seed}/inject_defects.txt
done
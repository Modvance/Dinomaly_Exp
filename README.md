### Original project address：https://github.com/guojiajeremy/Dinomaly

示例命令：

```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/macro/ --save_name warmup_pareto_1 --warmup_diag --warmup_milestones 50,100,150,200,250,300,350,400,450,500 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --gpus 0
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed03 --save_dir ../saved_results/turn/ --save_name warmup_pareto_1 --warmup_diag --warmup_milestones 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed03/inject_defects.txt --gpus 1
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed05 --save_dir ../saved_results/turn/ --save_name warmup_k4_1 --warmup_diag --warmup_milestones 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed05/inject_defects.txt --gpus 2
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../saved_results/turn/ --save_name warmup_k1_1 --warmup_diag --warmup_milestones 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt --gpus 4
```

```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed03 --save_dir ../saved_results/turn/ --save_name denose_pareto --gpus 1 --warmup_denoise --warmup_end_iter 90 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed03/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed05 --save_dir ../saved_results/turn/ --save_name denose_step_k4 --gpus 2 --warmup_denoise --warmup_end_iter 80 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed05/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../saved_results/turn/ --save_name denose_step_k1 --gpus 4 --warmup_denoise --warmup_end_iter 90 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt --warmup_save_scores_before_prune

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/turn/ --save_name denose_pareto --gpus 1 --warmup_denoise --warmup_end_iter 120 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed03/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/turn/ --save_name denose_step_k4 --gpus 2 --warmup_denoise --warmup_end_iter 150 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed05/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/turn/ --save_name denose_step_k1 --gpus 4 --warmup_denoise --warmup_end_iter 110 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt --warmup_save_scores_before_prune

```

```bash
python dinomaly_visa_uni.py --data_path ../visa_pytorch/1cls --save_dir ../saved_results/visa/ --save_name clean_visa --gpus 1
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-pareto-seed01 --save_dir ../saved_results/visa/ --save_name visa_warm_pareto --warmup_diag --warmup_milestones 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt --gpus 1
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k4-seed01 --save_dir ../saved_results/visa/ --save_name visa_warm_k4 --warmup_diag --warmup_milestones 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt --gpus 2
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k1-seed01 --save_dir ../saved_results/visa/ --save_name visa_warm_k1 --warmup_diag --warmup_milestones 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt --gpus 3

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/macro/ --save_name warmup_pareto_2 --warmup_diag --warmup_milestones 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --gpus 6
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/turn/ --save_name warmup_k4_1 --warmup_diag --warmup_milestones 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --gpus 2
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/turn/ --save_name warmup_k1_1 --warmup_diag --warmup_milestones 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --gpus 4
```

```bash
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-pareto-seed01 --save_dir ../saved_results/visa/ --save_name denose_pareto --gpus 1 --warmup_denoise --warmup_end_iter 360 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --warmup_save_scores_before_prune --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k4-seed01 --save_dir ../saved_results/visa/ --save_name denose_k4 --gpus 3 --warmup_denoise --warmup_end_iter 550 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --warmup_save_scores_before_prune --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k1-seed01 --save_dir ../saved_results/visa/ --save_name denose_k1_1 --gpus 4 --warmup_denoise --warmup_end_iter 380 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 5 --warmup_save_scores_before_prune --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt
```

```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/Postprocessing/ --save_name mvtec_pareto --gpus 1 --warmup_postprocess_mode soft --warmup_end_iter 120 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/Postprocessing/ --save_name mvtec_k4 --gpus 2 --warmup_postprocess_mode soft --warmup_end_iter 150 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/Postprocessing/ --save_name mvtec_k1 --gpus 3 --warmup_postprocess_mode soft --warmup_end_iter 110 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --warmup_save_scores_before_prune

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/Postprocessing/ --save_name hybrid_pareto --gpus 4 --warmup_postprocess_mode hybrid --warmup_end_iter 120 --warmup_hybrid_prune_ratio 0.05 --warmup_min_keep_per_class 5 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/Postprocessing/ --save_name hybrid_k4 --gpus 5 --warmup_postprocess_mode hybrid --warmup_end_iter 150 --warmup_hybrid_prune_ratio 0.05 --warmup_min_keep_per_class 5 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/Postprocessing/ --save_name hybrid_k1 --gpus 6 --warmup_postprocess_mode hybrid --warmup_end_iter 110 --warmup_hybrid_prune_ratio 0.05 --warmup_min_keep_per_class 5 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --warmup_save_scores_before_prune
```

tailedcore datasets
```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase2/ --save_name mvtec --gpus 1 --warmup_auto_end --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase2/ --save_name k4 --gpus 1 --warmup_auto_end --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase2/ --save_name k1 --gpus 1 --warmup_auto_end --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase2/ --save_name pareto_peak_dt_1 --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.05 --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 200
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase2/ --save_name k4_peak_dt_1 --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.05 --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 200
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase2/ --save_name k1_peak_dt_1 --gpus 4 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.05 --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 200

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase2/ --save_name pareto_peak_dt_global --gpus 1 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --warmup_dynamic_denoise --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_weight_scope global
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase2/ --save_name k4_peak_dt_global --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --warmup_dynamic_denoise --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_weight_scope global
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase2/ --save_name k1_peak_dt_global --gpus 6 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --warmup_dynamic_denoise --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_weight_scope global

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase2/ --save_name pareto_0 --warmup_diag --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --gpus 2
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase2/ --save_name k4_0 --warmup_diag --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --gpus 3
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase2/ --save_name k1_0 --warmup_diag --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --gpus 4

python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-pareto-seed01 --save_dir ../saved_results/phase2/ --save_name visa_pareto_0 --warmup_diag --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt --gpus 4
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k4-seed01 --save_dir ../saved_results/phase2/ --save_name visa_k4_0 --warmup_diag --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt --gpus 5
python dinomaly_visa_uni.py --data_path ../LTN_visa/visa-step_k1-seed01 --save_dir ../saved_results/phase2/ --save_name visa_k1_0 --warmup_diag --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt --gpus 6
```

0.1 overlap noisy datasets
```bash
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_0 --warmup_diag --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --gpus 1
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_hard --gpus 2 --warmup_postprocess_mode remove --warmup_end_iter 150 --warmup_prune_ratio 0.10 --warmup_min_keep_per_class 20 --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_soft --gpus 3 --warmup_postprocess_mode soft --warmup_end_iter 150 --warmup_weight_beta 1.0 --warmup_min_weight 0.2 --warmup_weight_scope class --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_save_scores_before_prune
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_auto --gpus 1 --warmup_auto_end --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_dt --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_gap --gpus 3 --warmup_auto_end --warmup_primary_metric score_gap --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_gap_1 --gpus 3 --warmup_auto_end --warmup_primary_metric score_gap --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_weight_beta 2.0 --warmup_min_weight 0.05 --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 200
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt --gpus 4 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_gap --gpus 2 --warmup_auto_end --warmup_primary_metric score_gap --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_2 --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.05 --denoise_weight_ema 0 --denoise_weight_clip_delta 0 --denoise_refresh_interval 200
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_0.5j --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_jaccard_threshold 0.5

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_weight_1 --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.2
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_weight_2 --gpus 4 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 1.0 --warmup_min_weight 0.1
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_weight_3 --gpus 1 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.1
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_weight_4 --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_weight_5 --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 2.0 --warmup_min_weight 0.05

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_refresh_1 --gpus 4 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.1 --denoise_refresh_interval 500
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_refresh_2 --gpus 5 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 500
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_refresh_3 --gpus 6 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --denoise_weight_ema 0.5 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_refresh_4 --gpus 1 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_refresh_5  --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.3 --denoise_refresh_interval 300

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_ema_1 --gpus 1 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_reliability_ema 0.9
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_ema_2 --gpus 2 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_reliability_ema 0.7
python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_ema_3 --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_reliability_ema 0.5

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_best --gpus 5 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_global --gpus 3 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_weight_beta 3.0 --warmup_min_weight 0.1 --warmup_dynamic_denoise --denoise_weight_ema 0.0 --denoise_weight_clip_delta 0.2 --denoise_refresh_interval 300 --denoise_weight_scope global

python dinomaly_mvtec_uni.py --data_path ../noisy_datasets/seed0 --save_dir ../saved_results/noisy/ --save_name mvtec_peak_dt_hard --gpus 5 --warmup_auto_end --warmup_primary_metric D_t --diag_manifest_path ../noisy_datasets/seed0/inject_defects.txt --warmup_trigger_mode peak_patience --warmup_postprocess_mode remove
```

eval
```bash
python dinomaly_mvtec_uni.py --data_path ../mvtec_anomaly_detection --save_dir ../saved_results/eval/ --save_name mvtec --gpus 3 --eval_interval 500
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/eval/ --save_name mvtec_pareto --gpus 0 --eval_interval 500
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/eval/ --save_name mvtec_k4 --gpus 2 --eval_interval 500
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/eval/ --save_name mvtec_k1 --gpus 3 --eval_interval 500
```

gbps
```bash
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_pareto --gpus 1 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_k4 --gpus 5 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_k1 --gpus 6 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_k1_gp15 --gpus 1 --gbps_num_groups 15 --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt

eval:
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_pareto_eval --gpus 1 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --eval_interval 500
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_k4_eval --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --eval_interval 500
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/gbps/ --save_name mvtec_k1_eval --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --eval_interval 500

lt:
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/lt --save_name pareto --gpus 1 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable --lt_group_balanced_loss --lt_tail_aug_enable
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/lt --save_name k4 --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable --lt_group_balanced_loss --lt_tail_aug_enable
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/lt --save_name k1 --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable --lt_group_balanced_loss --lt_tail_aug_enable

sampler:
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/lt --save_name pareto_spl --gpus 1 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/lt --save_name k4_spl --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/lt --save_name k1_spl --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_sampler_enable

loss:
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/lt --save_name pareto_loss --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_group_balanced_loss
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/lt --save_name k4_loss --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_group_balanced_loss
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/lt --save_name k1_loss --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --lt_enable --lt_group_balanced_loss
```

patch
```bash
python dinomaly_mvtec_patch.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/patch/ --save_name mvtec_pareto --gpus 1
python dinomaly_mvtec_patch.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/patch/ --save_name mvtec_k4 --gpus 2
python dinomaly_mvtec_patch.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/patch/ --save_name mvtec_k1 --gpus 3
```

phase5
```bash
python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase5 --save_name mvtec_pareto --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt --gpus 1
python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase5 --save_name mvtec_k4 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed02/inject_defects.txt --gpus 2
python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase5 --save_name mvtec_k1 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed02/inject_defects.txt --gpus 3

CUDA_VISIBLE_DEVICES=1 python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase5/ --save_name mvtec_pareto --gpus 0 --phase5_use_faiss 1 --phase5_faiss_gpu 1 --phase5_faiss_gpu_id 0 --phase5_knn_backend faiss --phase5_kmeans_backend faiss --phase5_force_recompute
CUDA_VISIBLE_DEVICES=2 python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-step_k4-seed02 --save_dir ../saved_results/phase5/ --save_name mvtec_k4 --gpus 0 --phase5_use_faiss 1 --phase5_faiss_gpu 1 --phase5_faiss_gpu_id 0 --phase5_knn_backend faiss --phase5_kmeans_backend faiss --phase5_force_recompute
CUDA_VISIBLE_DEVICES=3 python dinomaly_mvtec_phase5.py --data_path ../LTN_datasets/mvtecad-step_k1-seed02 --save_dir ../saved_results/phase5/ --save_name mvtec_k1 --gpus 0 --phase5_use_faiss 1 --phase5_faiss_gpu 1 --phase5_faiss_gpu_id 0 --phase5_knn_backend faiss --phase5_kmeans_backend faiss --phase5_force_recompute
```

class
```bash
python run_precluster_visual.py --image_root ../mvtec_anomaly_detection --output_dir ../results/class/vlm_1 --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-pareto-seed02 --output_dir ../results/class/vlm_pareto_1 --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-step_k4-seed02 --output_dir ../results/class/vlm_k4 --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-step_k1-seed02 --output_dir ../results/class/vlm_k1 --stage all --split train --device cuda

python run_precluster_visual.py --image_root ../mvtec_anomaly_detection --output_dir ../results/class/vlm_proto --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-pareto-seed02 --output_dir ../results/class/vlm_proto_pareto --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-step_k4-seed02 --output_dir ../results/class/vlm_proto_k4 --stage all --split train --device cuda
python run_precluster_visual.py --image_root ../LTN_datasets/mvtecad-step_k1-seed02 --output_dir ../results/class/vlm_proto_k1 --stage all --split train --device cuda
```

test
```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed00 --save_dir ../results/test --save_name dino_mvtec_k4_clean --gpus 1
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name dino_mvtec_k4 --gpus 2
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name gbps_mvtec_k4 --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name lt_mvtec_k4_spl --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt --lt_enable --lt_sampler_enabl
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name lt_mvtec_k4_loss --gpus 1 --gbps_auto_k --gbps_postprocess_mode remove  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt --lt_enable --lt_group_balanced_loss
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name lt_mvtec_k4_spl_loss --gpus 5 --gbps_auto_k --gbps_postprocess_mode remove  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt --lt_enable --lt_sampler_enable --lt_group_balanced_loss
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name lt_mvtec_k4_spl_loss_aug --gpus 6 --gbps_auto_k --gbps_postprocess_mode remove  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt --lt_enable --lt_sampler_enable --lt_group_balanced_loss --lt_tail_aug_enable

python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_clean --gpus 1
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/test --save_name dino_mvtec_k1 --gpus 2
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/test --save_name gbps_mvtec_k1 --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_spl --gpus 1 --lt_sampler_enable
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_loss --gpus 2 --lt_group_balanced_loss
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_spl_loss_aug --gpus 3 --lt_sampler_enable --lt_group_balanced_loss --lt_tail_aug_enable
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_spl_aug --gpus 2 --lt_sampler_enable --lt_tail_aug_enable
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_spl_0.3_6 --gpus 1 --lt_sampler_enable --lt_sampler_enable --lt_sampler_alpha 0.3 --lt_sampler_clip_max 6
python dinomaly_mvtec_phase6.py --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_mvtec_k1_clean_spl_0.1_8 --gpus 3 --lt_sampler_enable --lt_sampler_enable --lt_sampler_alpha 0.1 --lt_sampler_clip_max 8
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/test --save_name gbps_mvtec_k1_spl_0.3_6 --gpus 3 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt --lt_sampler_enable --lt_sampler_enable --lt_sampler_alpha 0.3 --lt_sampler_clip_max 6
python dinomaly_mvtec_gbps.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/test --save_name gbps_mvtec_k4_spl_0.3_6 --gpus 2 --gbps_auto_k --gbps_postprocess_mode remove --gbps_prune_ratio 0.1 --gbps_min_keep_per_group 8 --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt --lt_sampler_enable --lt_sampler_enable --lt_sampler_alpha 0.3 --lt_sampler_clip_max 6

python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/dino_mvtec_k1_clean/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_mem_s8_top100_lam1 --mem_enable --mem_apply_mode class_size_le --mem_class_size_thr 8 --mem_score_mode fused --mem_topk_ratio 1 --gpus 1 --mem_fusion_lambda 1.0
python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/dino_mvtec_k1_clean/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_mem_s8_top10_lam1 --mem_enable --mem_apply_mode class_size_le --mem_class_size_thr 8 --mem_score_mode fused --mem_topk_ratio 0.1 --gpus 2 --mem_fusion_lambda 1.0
python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/dino_mvtec_k1_clean/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_mem_s8_top5_lam1 --mem_enable --mem_apply_mode class_size_le --mem_class_size_thr 8 --mem_score_mode fused --mem_topk_ratio 0.05 --gpus 3 --mem_fusion_lambda 1.0
python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/dino_mvtec_k1_clean/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_mem_s8_top1_lam1 --mem_enable --mem_apply_mode class_size_le --mem_class_size_thr 8 --mem_score_mode fused --mem_topk_ratio 0.01 --gpus 3 --mem_fusion_lambda 1.0
python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/dino_mvtec_k1_clean/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name dino_mvtec_k1_mem_all_top5_lam1 --mem_enable --mem_apply_mode all --mem_score_mode fused --mem_topk_ratio 0.05 --gpus 2 --mem_fusion_lambda 1.0
python dinomaly_mvtec_one_shot_memory_eval.py --checkpoint_path ../results/test/lt_mvtec_k1_clean_spl_0.3_6/final_model.pt --data_path ../LTN_datasets/mvtecad-step_k1-seed00 --save_dir ../results/test --save_name lt_spl_mvtec_k1_mem_s8_top5_lam1 --mem_enable --mem_apply_mode class_size_le --mem_class_size_thr 8 --mem_score_mode fused --mem_topk_ratio 0.05 --gpus 2 --mem_fusion_lambda 1.0
```

TailedCore
```bash
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-pareto-seed01 --save_dir ../results/tailedcore --save_name dino_pareto_2 --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-pareto-seed01 --save_sampler_details --gpus 1
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/tailedcore --save_name dino_k4_2 --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-step_k4-seed01 --save_sampler_details --gpus 2
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/tailedcore --save_name dino_k1_2 --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-step_k1-seed01 --save_sampler_details --gpus 3

python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-pareto-seed01 --save_dir ../results/tailedcore --save_name dino_cls_pareto --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-pareto-seed01 --tailsampler_embedding_source encoder_cls --save_sampler_details --gpus 1
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/tailedcore --save_name dino_cls_k4 --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-step_k4-seed01 --tailsampler_embedding_source encoder_cls --save_sampler_details --gpus 2
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/tailedcore --save_name dino_cls_k1 --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --tailsampler_dataset_name mvtecad-step_k1-seed01 --tailsampler_embedding_source encoder_cls --save_sampler_details --gpus 3

python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-pareto-seed01 --save_dir ../results/tailedcore --save_name dino_cls_hrt_pareto_1 --tailsampler_embedding_source encoder_cls --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --save_sampler_details --enable_hrt --hrt_embedding_source encoder_cls --hrt_group_k 7 --hrt_head_topm 20 --hrt_head_topk_mean 5 --hrt_trim_lambda 2.5 --hrt_min_keep_ratio 0.70 --hrt_max_drop_ratio 0.30 --hrt_save_patch_maps --gpus 1
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k4-seed01 --save_dir ../results/tailedcore --save_name dino_cls_hrt_k4_1 --tailsampler_embedding_source encoder_cls --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --save_sampler_details --enable_hrt --hrt_embedding_source encoder_cls --hrt_group_k 7 --hrt_head_topm 20 --hrt_head_topk_mean 5 --hrt_trim_lambda 2.5 --hrt_min_keep_ratio 0.70 --hrt_max_drop_ratio 0.30 --hrt_save_patch_maps --gpus 2
python dinomaly_mvtec_tailsampler.py --data_path ../LTN_datasets/mvtecad-step_k1-seed01 --save_dir ../results/tailedcore --save_name dino_cls_hrt_k1_1 --tailsampler_embedding_source encoder_cls --tailsampler_type adaptive_trim_mode --tailsampler_gt_mode dataset_rule --save_sampler_details --enable_hrt --hrt_embedding_source encoder_cls --hrt_group_k 7 --hrt_head_topm 20 --hrt_head_topk_mean 5 --hrt_trim_lambda 2.5 --hrt_min_keep_ratio 0.70 --hrt_max_drop_ratio 0.30 --hrt_save_patch_maps --gpus 3
```

## TailGuard（唯一论文主线）

当前论文方法统一命名为 **TailGuard**。历史名称中的 `B` 只是开发期临时标记，不再表示版本；新实验、日志和论文均不应再使用 TailGuard-B、`tgb_*` 或 `tailguard_b/`。

主入口：

- MVTec：`dinomaly_mvtec_tailguard.py`
- VisA：`dinomaly_visa_tailguard.py`
- attachment replay：`dinomaly_mvtec_tailguard_attachment_replay.py`
- memory rebuild：`dinomaly_mvtec_tailguard_memory_rebuild.py`

不提供任何方法参数时，runner 就使用论文最终配置：`adaptive_trim_mode`、H-group auto-K、`all_samples` Stage 1、H-only Safe-GBPS、稳定 active group 内的自适应删除比例（10% 安全上限）、RGD + segmented-BIC，以及 pseudo-class tail memory。README 同时承担实验命令记录，因此下面仍显式写出关键方法参数，便于以后直接核对日志和复现实验。

以下命令默认在 `/home/linux/projects/Dinomaly/` 下、已激活 `dinomaly` 环境后执行。

### MVTec

```bash
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard \
  --save_name seg_pareto \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 1

python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard \
  --save_name seg_k4 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 2

python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard \
  --save_name seg_k1 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 3
```

### VisA

```bash
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/visa \
  --save_name seg_pareto \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 0

python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/visa \
  --save_name seg_k4 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 1

python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/visa \
  --save_name seg_k1 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_memory_enable \
  --gpus 2
```

每次运行的主产物位于 `<save_dir>/<save_name>/tailguard/`，最终模型位于 `<save_dir>/<save_name>/final_model.pt`。

H 清理有两条显式可选路径：

- `--gbps_prune_mode adaptive`：默认主线。仅稳定激活的 H group 参与删除，请求比例为 `min(pi_high, gbps_prune_max_ratio)`；默认安全上限为 `0.10`，实际删除数还受向下取整和 `gbps_min_keep_per_group` 约束，因此组可以少删或不删。
- `--gbps_prune_mode fixed`：固定比例对照。保持历史逻辑，在每个 H group 内请求删除最高分的 `gbps_prune_ratio`；默认是 `0.10`，实际删除数同样受向下取整和最小保留数约束。

固定 10% 消融只需在任一训练命令中替换为：

```bash
--gbps_prune_mode fixed \
--gbps_prune_ratio 0.10
```

新运行会在 `tailguard/stage2/` 保存 `h_prune_decisions.csv`、`h_prune_group_summary.csv` 和 `h_prune_summary.json`，记录每组估计比例、稳定性、请求/实际删除数与安全上限是否生效。

### MVTec 离线 replay 与 memory rebuild

下面每组命令先从训练运行重建 attachment/pseudo-class，再使用该 replay 输出重建 memory。

```bash
# Pareto
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir ../results/tailguard/seg_pareto/tailguard \
  --output_dir ../results/tailguard/seg_pareto_attachment_replay

python dinomaly_mvtec_tailguard_memory_rebuild.py \
  --checkpoint_path ../results/tailguard/seg_pareto/final_model.pt \
  --pseudoclass_dir ../results/tailguard/seg_pareto_attachment_replay/pseudoclasses \
  --output_dir ../results/tailguard/seg_pareto_memory_rebuild \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --gpu 1

# Step-K4
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir ../results/tailguard/seg_k4/tailguard \
  --output_dir ../results/tailguard/seg_k4_attachment_replay

python dinomaly_mvtec_tailguard_memory_rebuild.py \
  --checkpoint_path ../results/tailguard/seg_k4/final_model.pt \
  --pseudoclass_dir ../results/tailguard/seg_k4_attachment_replay/pseudoclasses \
  --output_dir ../results/tailguard/seg_k4_memory_rebuild \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --gpu 0

# Step-K1
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir ../results/tailguard/seg_k1/tailguard \
  --output_dir ../results/tailguard/seg_k1_attachment_replay

python dinomaly_mvtec_tailguard_memory_rebuild.py \
  --checkpoint_path ../results/tailguard/seg_k1/final_model.pt \
  --pseudoclass_dir ../results/tailguard/seg_k1_attachment_replay/pseudoclasses \
  --output_dir ../results/tailguard/seg_k1_memory_rebuild \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --gpu 1
```

新命令只使用 `tg_*`。工具仍可读取旧 checkpoint/config 中的 `tgb_*` 键，但这些名称仅用于兼容历史产物。

历史实验的物理目录（例如 `../results/tailguard_b/rgd_k4/tailguard_b`）不应重命名，否则已有日志和 artifact 中的路径会失效。读取它们时使用新工具并明确标注为历史结果，例如：

```bash
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir ../results/tailguard_b/rgd_k4/tailguard_b \
  --output_dir ../results/tailguard/legacy_rgd_k4_attachment_replay
```

历史 attachment 消融也统一使用新入口和 `tg_*` 参数，但源目录继续指向真实的旧 artifact：

```bash
# H-calibrated attachment
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir /home/linux/projects/results/tailguard_b/k4/tailguard_b \
  --output_dir ../results/tailguard/ablations/k4_h_calibrated \
  --tg_attachment_membership_mode h_calibrated

# RGD + largest-positive-gap 历史切分消融
python dinomaly_mvtec_tailguard_attachment_replay.py \
  --source_tailguard_dir /home/linux/projects/results/tailguard_b/k4/tailguard_b \
  --output_dir ../results/tailguard/ablations/k4_rgd_largest_gap \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode largest_positive_gap
```

更完整的默认配置、消融覆盖方式和产物说明见 `../docs/use.md`。

## TGM（历史消融，非论文主线）

原先名为 TailGuard 的早期全局组风险与 group-memory 方法现统一重命名为 **TGM**。它只用于历史消融，不代表当前 TailGuard，也不应出现在论文主方法命令中。

入口为 `dinomaly_mvtec_tgm.py`，专用参数前缀为 `tgm_*`：

```bash
python dinomaly_mvtec_tgm.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tgm \
  --save_name tgm_pareto \
  --tailsampler_type adaptive_trim_mode \
  --tgm_tail_embedding_source encoder_cls \
  --gbps_auto_k \
  --gbps_postprocess_mode remove \
  --tgm_memory_enable \
  --gpus 1

python dinomaly_mvtec_tgm.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tgm \
  --save_name tgm_k4 \
  --tailsampler_type adaptive_trim_mode \
  --tgm_tail_embedding_source encoder_cls \
  --gbps_auto_k \
  --gbps_postprocess_mode remove \
  --tgm_memory_enable \
  --gpus 2

python dinomaly_mvtec_tgm.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tgm \
  --save_name tgm_k1 \
  --tailsampler_type adaptive_trim_mode \
  --tgm_tail_embedding_source encoder_cls \
  --gbps_auto_k \
  --gbps_postprocess_mode remove \
  --tgm_memory_enable \
  --gpus 3
```

## TailGuard Adaptive 主线拉表实验（6 组）

以下六条命令用于论文主表：MVTec AD 和 VisA 各运行 Pareto、Step-K4、Step-K1，统一使用 `seed01`。它们使用当前 TailGuard schema v3 最终主线配置：稳定 active H group、按 `pi_high` 自适应确定请求比例、每组 10% 安全上限、RGD + segmented-BIC，以及最终 pseudo-class memory 评估。

六个实验使用独立的 `save_name`，统一写入 `../results/tailguard/main_table_adaptive/`，不会覆盖上面的历史命令。下面预分配 GPU 1–6，若不并行运行，只需按机器情况修改 `--gpus`。每条命令完整训练 `10000` iterations，并同时保存最终 reconstruction 指标和 TailGuard memory-fused 指标。

### MVTec AD：Pareto / Step-K4 / Step-K1

```bash
# 1. MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name mvtec_pareto_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 2. MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name mvtec_step_k4_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 3. MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name mvtec_step_k1_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3
```

### VisA：Pareto / Step-K4 / Step-K1

运行 VisA 前应确认 `../LTN_visa/visa-*-seed01` 已生成或挂载；manifest 使用仓库内的 `manifest/visa-nlt/`。

```bash
# 4. VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name visa_pareto_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4

# 5. VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name visa_step_k4_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 5

# 6. VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/main_table_adaptive \
  --save_name visa_step_k1_seed01 \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 6
```

每个实验完成后，主表所需汇总可从以下位置读取：

- reconstruction 最终指标：`<save_dir>/<save_name>/tailguard/tailguard_summary.json` 中的 `final_eval_summary`；
- TailGuard 主线 memory-fused 指标：`<save_dir>/<save_name>/tailguard/memory/memory_eval_summary.json`；
- 最终模型：`<save_dir>/<save_name>/final_model.pt`；
- 自适应 H 清理明细：`<save_dir>/<save_name>/tailguard/stage2/h_prune_summary.json` 和对应 CSV。

拉表前的完成验收以 `tailguard_summary.json` 中 `memory_status == "completed"` 为准；只生成 `final_model.pt` 说明主模型已保存，并不一定表示 memory-fused 评估成功完成。`memory_eval_summary.json` 的 `metrics_by_mode` 包含 `recon`、`memory`、`fused` 三种模式的总体及逐类别七项指标，主表应读取 `fused`。

当前工作区核对结果：三组 MVTec AD 数据和六份 manifest 均存在；VisA manifest 已存在，但 `../LTN_visa/visa-*-seed01` 三个数据目录尚未生成或挂载，因此 VisA 三条命令当前会在启动检查时报错。可先根据 `make_all_visa_nlt.sh` 生成数据，再启动第 4–6 组实验。

## TailGuard 核心模块消融实验

核心模块消融采用累加式设计，只验证会实际改变最终异常检测分数的三个模块：

| 行 | H 自适应清理 | RGD + attached-T 风险清理 | pseudo-class memory | 指标来源 |
|---|---:|---:|---:|---|
| A0 Base |  |  |  | 本节 A0 运行的 `metrics_by_mode.recon.summary` |
| A1 + H Cleanup | ✓ |  |  | 本节 A1 运行的 `metrics_by_mode.recon.summary` |
| A2 + T Refinement | ✓ | ✓ |  | 对应完整主线运行的 `metrics_by_mode.recon.summary` |
| A3 Full TailGuard | ✓ | ✓ | ✓ | 对应完整主线运行的 `metrics_by_mode.fused.summary` |

A0 和 A1 仍保持 `--gbps_postprocess_mode remove`、selected-checkpoint 恢复、Stage 2 重建以及最终 memory 评估。这样四行可以统一从 `memory_eval_summary.json` 的同一套 evaluator 读取指标；A0/A1 虽然会在训练结束后构建 memory，但论文取值只读取 `recon`，memory 不参与这两行的最终分数。

下面先在两个代表性困难场景运行：MVTec AD Step-K1 seed01 和 VisA Step-K1 seed01。四条命令使用独立 `save_name`，统一写入 `../results/tailguard/module_ablation/`，不会覆盖主线结果。GPU 1–4 只是示例分配，实际运行时按机器空闲情况修改。

### A0：Base（不删除任何 H/T 样本）

```bash
# 1. MVTec AD / Step-K1 / A0 Base / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k1_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 2. VisA / Step-K1 / A0 Base / seed01 / GPU 2
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k1_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2
```

A0 完成后应检查：

```text
<run>/tailguard/stage2/stage2_dataset_summary.json
num_stage2_removed == 0
```

### A1：加入 H 自适应清理，但不删除 attached-T

```bash
# 3. MVTec AD / Step-K1 / A1 H Cleanup / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k1_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 4. VisA / Step-K1 / A1 H Cleanup / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k1_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4
```

A1 完成后应检查：

```text
<run>/tailguard/stage2/h_prune_summary.json
num_pruned > 0

<run>/tailguard/stage2/stage2_dataset_summary.json
num_tail_head_noise == 0
```

### A2 与 A3：直接复用完整主线运行

A2 和 A3 不需要重新训练。统一从对应 Step-K1 主线结果读取：

```text
# MVTec AD Step-K1
../results/tailguard/main_table_adaptive/mvtec_step_k1_seed01/tailguard/memory/memory_eval_summary.json

# VisA Step-K1
../results/tailguard/main_table_adaptive/visa_step_k1_seed01/tailguard/memory/memory_eval_summary.json
```

取值规则：

```text
A0 Base              -> A0 文件的 metrics_by_mode.recon.summary
A1 + H Cleanup       -> A1 文件的 metrics_by_mode.recon.summary
A2 + T Refinement    -> Full 文件的 metrics_by_mode.recon.summary
A3 Full TailGuard    -> Full 文件的 metrics_by_mode.fused.summary
```

所有 A0/A1/Full 运行均以各自 `tailguard/tailguard_summary.json` 中 `memory_status == "completed"` 作为完成验收条件。若 A0 的删除数不为 0，或 A1 仍出现 `num_tail_head_noise > 0`，不要直接拉表，应先核对实际 resolved config。

### 扩展到 Pareto 与 Step-K4（剩余 8 条命令）

以下命令补齐 MVTec AD 和 VisA 的 Pareto、Step-K4 组件消融。加上前面的四条 Step-K1 命令后，A0/A1 将覆盖全部六个主线场景；A2/A3 仍直接复用对应完整主线运行的 `recon`/`fused` 指标。

下面按两批 GPU 1–4 编排：先运行四条 A0，再运行四条 A1。若需要跨批并行，请自行改成空闲 GPU，避免同卡冲突。

#### A0 Base：Pareto / Step-K4

```bash
# 5. MVTec AD / Pareto / A0 Base / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_pareto_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 6. MVTec AD / Step-K4 / A0 Base / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k4_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 7. VisA / Pareto / A0 Base / seed01 / GPU 3
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_pareto_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 8. VisA / Step-K4 / A0 Base / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k4_seed01_a0_base \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4
```

#### A1 + H Cleanup：Pareto / Step-K4

```bash
# 9. MVTec AD / Pareto / A1 H Cleanup / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_pareto_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 10. MVTec AD / Step-K4 / A1 H Cleanup / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k4_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 11. VisA / Pareto / A1 H Cleanup / seed01 / GPU 3
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_pareto_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 12. VisA / Step-K4 / A1 H Cleanup / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k4_seed01_a1_h_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4
```

六个场景的 A2/A3 完整主线结果路径如下：

```text
../results/tailguard/main_table_adaptive/mvtec_pareto_seed01/tailguard/memory/memory_eval_summary.json
../results/tailguard/main_table_adaptive/mvtec_step_k4_seed01/tailguard/memory/memory_eval_summary.json
../results/tailguard/main_table_adaptive/mvtec_step_k1_seed01/tailguard/memory/memory_eval_summary.json
../results/tailguard/main_table_adaptive/visa_pareto_seed01/tailguard/memory/memory_eval_summary.json
../results/tailguard/main_table_adaptive/visa_step_k4_seed01/tailguard/memory/memory_eval_summary.json
../results/tailguard/main_table_adaptive/visa_step_k1_seed01/tailguard/memory/memory_eval_summary.json
```

对每个场景均使用同一取值规则：A0/A1 读取各自 `metrics_by_mode.recon.summary`，A2 读取对应 Full 的 `metrics_by_mode.recon.summary`，A3 读取对应 Full 的 `metrics_by_mode.fused.summary`。

## TailGuard 完整组合消融：剩余实验

本节只补充前述实验矩阵中尚未覆盖的运行，不替换已有命令：

- Original Dinomaly：严格使用原始 `dinomaly_*_uni.py` 连续训练，作为独立外部基线；
- S010/S011 T-only：关闭 H 样本清理，仅开启 attached-T 风险清理；同一次运行分别读取 `recon` 和 `fused`，无需重复训练。

三个消融因子定义为：H 表示 H 自适应样本清理，T 表示 attached-T 风险样本清理，M 表示 pseudo-class memory 分数融合。RGD attachment 是所有 TailGuard scaffold 配置共享的内部流程，不属于这里的 T 开关。

完整取值关系如下：

| 配置 | H | T | M | 运行来源 | 指标来源 |
|---|---:|---:|---:|---|---|
| Original Dinomaly | - | - | - | 本节 Original 命令 | `final_model.pt` 中的 `final_eval_summary` |
| S000 Scaffold | 0 | 0 | 0 | 前述 A0 | `metrics_by_mode.recon.summary` |
| S001 Scaffold + Memory | 0 | 0 | 1 | 前述 A0 | `metrics_by_mode.fused.summary` |
| S100 H-only | 1 | 0 | 0 | 前述 A1 | `metrics_by_mode.recon.summary` |
| S101 H-only + Memory | 1 | 0 | 1 | 前述 A1 | `metrics_by_mode.fused.summary` |
| S010 T-only | 0 | 1 | 0 | 本节 T-only | `metrics_by_mode.recon.summary` |
| S011 T-only + Memory | 0 | 1 | 1 | 本节 T-only | `metrics_by_mode.fused.summary` |
| S110 H + T | 1 | 1 | 0 | 前述完整主线 | `metrics_by_mode.recon.summary` |
| S111 Full TailGuard | 1 | 1 | 1 | 前述完整主线 | `metrics_by_mode.fused.summary` |

下面仍按 GPU 1-6 给出示例分配。若分批或与其他实验同时运行，应按实际空闲 GPU 调整，避免同卡冲突。当前三组 VisA 数据目录尚未生成或挂载，因此所有 VisA 命令需在准备 `../LTN_visa/visa-*-seed01` 后执行。

### Original Dinomaly（6 个场景）

以下命令不启用任何 TailGuard、warmup、诊断或去噪功能；`--warmup_postprocess_mode none` 用于显式锁定无后处理配置。`dinomaly_mvtec_uni.py` 和 `dinomaly_visa_uni.py` 在代码中固定使用 seed 1、batch size 16 和 10000 iterations，因此不支持也不需要 `--total_iters`。

```bash
# 1. Original Dinomaly / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_uni.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name mvtec_pareto_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 1

# 2. Original Dinomaly / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_uni.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name mvtec_step_k4_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 2

# 3. Original Dinomaly / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_uni.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name mvtec_step_k1_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 3

# 4. Original Dinomaly / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_uni.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name visa_pareto_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 4

# 5. Original Dinomaly / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_uni.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name visa_step_k4_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 5

# 6. Original Dinomaly / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_uni.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --save_dir ../results/tailguard/original_dinomaly_baseline \
  --save_name visa_step_k1_seed01_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 6
```

每个 Original Dinomaly 实验完成后，从以下 checkpoint 的 `final_eval_summary` 读取七项 reconstruction 指标，并确认 `iteration == 10000`：

```text
../results/tailguard/original_dinomaly_baseline/mvtec_pareto_seed01_original_dinomaly/final_model.pt
../results/tailguard/original_dinomaly_baseline/mvtec_step_k4_seed01_original_dinomaly/final_model.pt
../results/tailguard/original_dinomaly_baseline/mvtec_step_k1_seed01_original_dinomaly/final_model.pt
../results/tailguard/original_dinomaly_baseline/visa_pareto_seed01_original_dinomaly/final_model.pt
../results/tailguard/original_dinomaly_baseline/visa_step_k4_seed01_original_dinomaly/final_model.pt
../results/tailguard/original_dinomaly_baseline/visa_step_k1_seed01_original_dinomaly/final_model.pt
```

### S010/S011：T-only（6 个场景）

T-only 与完整主线保持相同训练协议，仅使用 `--gbps_prune_max_ratio 0.0` 关闭 H 删除，同时保留 `--tg_t_attached_noise_p_high_thr 0.50` 开启 attached-T 风险删除。必须保留 `--gbps_postprocess_mode remove`，否则不会进入 attachment、Stage 2 和 memory 分支。

```bash
# 1. T-only / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_pareto_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 2. T-only / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k4_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 3. T-only / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name mvtec_step_k1_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 4. T-only / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_pareto_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4

# 5. T-only / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k4_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 5

# 6. T-only / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/module_ablation \
  --save_name visa_step_k1_seed01_s010_t_cleanup \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 0.50 \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 6
```

六个 T-only 实验的指标文件如下。S010 读取 `metrics_by_mode.recon.summary`，S011 读取同一文件的 `metrics_by_mode.fused.summary`：

```text
../results/tailguard/module_ablation/mvtec_pareto_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
../results/tailguard/module_ablation/mvtec_step_k4_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
../results/tailguard/module_ablation/mvtec_step_k1_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
../results/tailguard/module_ablation/visa_pareto_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
../results/tailguard/module_ablation/visa_step_k4_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
../results/tailguard/module_ablation/visa_step_k1_seed01_s010_t_cleanup/tailguard/memory/memory_eval_summary.json
```

每个 T-only 实验完成后应同时检查：

```text
<run>/tailguard/tailguard_summary.json
memory_status == "completed"

<run>/tailguard/stage2/h_prune_summary.json
num_pruned == 0

<run>/tailguard/stage2/stage2_dataset_summary.json
num_h_removed == 0
num_stage2_removed == num_tail_head_noise
```

`num_tail_head_noise == 0` 不一定代表运行失败，但说明该场景中 T 清理没有实际删除样本。拉表时应同时记录该值，避免把未激活的 T 因子解释为有效清理。


## 2×2 动机因果控制：需要新增的 12 组 Original Dinomaly 实验

本节使用 Original Dinomaly 分离训练集长尾与训练污染的影响。完整设计包含两个数据集、三个场景和四种数据条件，共 24 个结果格：

| 数据条件 | 训练分布 | 训练污染 | 本节处理方式 |
|---|---|---|---|
| Balanced-Clean | 原始非长尾分布 | 无 | 复用已有 MVTec AD、VisA 干净 Original checkpoint；同一数据集只需一个 checkpoint |
| Balanced-Noisy | 原始非长尾分布 | 有 | 本节新增 6 次训练 |
| Long-Tail-Clean | Pareto / Step-K4 / Step-K1 | 无 | 本节新增 6 次训练 |
| Long-Tail-Noisy | Pareto / Step-K4 / Step-K1 | 有 | 复用前述 6 次长尾污染 Original Dinomaly 结果 |

因此本节只新增 `6 + 6 = 12` 次训练。Balanced-Clean 在同一数据集的 Pareto、Step-K4、Step-K1 三行中复用同一个干净 checkpoint，不要重复训练。Long-Tail-Noisy 继续读取前述 `../results/tailguard/original_dinomaly_baseline/` 下的六个 checkpoint。

所有新增实验均严格运行 Original Dinomaly：不传 `--diag_manifest_path`，不启用 TailGuard、诊断或去噪功能，并用 `--warmup_postprocess_mode none` 显式关闭后处理。代码固定训练 10000 iterations，因此不传 `--total_iters`。

### 先构建 seed01 的 2×2 数据目录

以下命令一次构建 MVTec AD、VisA 的 Pareto、Step-K4、Step-K1 四种数据条件。默认使用符号链接；运行训练前应确认源数据路径在当前机器上有效。

```bash
# 只检查源数据、manifest 和目标路径，不创建数据
python build_causal_2x2_datasets.py \
  --seed 1 \
  --scenario all \
  --dry-run

# 正式构建数据
python build_causal_2x2_datasets.py \
  --seed 1 \
  --scenario all
```

默认输出到：

```text
../causal_2x2_datasets/{mvtec,visa}/{pareto,step_k4,step_k1}/seed01/{balanced-clean,balanced-noisy,long-tail-clean,long-tail-noisy}
```

### 第一批：Balanced-Noisy（6 次）

每个 Balanced-Noisy 场景使用与对应 Long-Tail-Noisy 场景相同的污染 manifest，但不应用长尾裁剪。GPU 1-6 仅为示例分配。

```bash
# 1. Balanced-Noisy / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/pareto/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_pareto_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 1

# 2. Balanced-Noisy / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/step_k4/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_step_k4_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 2

# 3. Balanced-Noisy / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/step_k1/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_step_k1_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 3

# 4. Balanced-Noisy / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/pareto/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_pareto_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 4

# 5. Balanced-Noisy / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/step_k4/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_step_k4_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 5

# 6. Balanced-Noisy / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/step_k1/seed01/balanced-noisy \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_step_k1_seed01_balanced_noisy_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 6
```

### 第二批：Long-Tail-Clean（6 次）

第二批复用 GPU 1-6，因此应在第一批全部结束后运行；若需要两批并行，请改为实际空闲 GPU。每个 Long-Tail-Clean 场景应用对应 `prune_good.txt`，但其训练集不包含任何注入异常。

```bash
# 7. Long-Tail-Clean / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/pareto/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_pareto_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 1

# 8. Long-Tail-Clean / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/step_k4/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_step_k4_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 2

# 9. Long-Tail-Clean / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_uni.py \
  --data_path ../causal_2x2_datasets/mvtec/step_k1/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name mvtec_step_k1_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 3

# 10. Long-Tail-Clean / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/pareto/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_pareto_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 4

# 11. Long-Tail-Clean / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/step_k4/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_step_k4_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 5

# 12. Long-Tail-Clean / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_uni.py \
  --data_path ../causal_2x2_datasets/visa/step_k1/seed01/long-tail-clean \
  --save_dir ../results/tailguard/causal_2x2_original \
  --save_name visa_step_k1_seed01_long_tail_clean_original_dinomaly \
  --warmup_postprocess_mode none \
  --eval_interval 5000 \
  --gpus 6
```

### 完成验收与 24 格取值

每次新增训练完成后检查：

```text
../results/tailguard/causal_2x2_original/<save_name>/final_model.pt
```

读取 checkpoint 中的 `final_eval_summary`，并确认 `iteration == 10000`。最终 24 个结果格按以下规则取值：

| 结果格 | 指标来源 |
|---|---|
| MVTec AD / Balanced-Clean / 三个场景 | 同一个已有干净 MVTec AD Original checkpoint |
| VisA / Balanced-Clean / 三个场景 | 同一个已有干净 VisA Original checkpoint |
| Balanced-Noisy / 六个场景 | 本节第一批六个 checkpoint |
| Long-Tail-Clean / 六个场景 | 本节第二批六个 checkpoint |
| Long-Tail-Noisy / 六个场景 | 前述六个 `original_dinomaly_baseline` checkpoint |

计算因果效应时，应在同一数据集、同一场景内比较：

```text
Long-tail 主效应（Clean） = Long-Tail-Clean - Balanced-Clean
Noise 主效应（Balanced） = Balanced-Noisy - Balanced-Clean
Noise 主效应（Long-tail） = Long-Tail-Noisy - Long-Tail-Clean
交互效应 = (Long-Tail-Noisy - Long-Tail-Clean) - (Balanced-Noisy - Balanced-Clean)
```


## 论文新版依赖感知消融（最新规范，24 次训练）

> **以本节为准。** README 前面的 `A0/A1`、`S000--S111`、T 硬删除及旧组合消融仅保留为历史记录，不再用于论文，也不要继续执行。本节所有结果必须由当前代码重新训练，不能与旧 schema 的 Full 结果混用。

TailSampler 是四个运行共享的候选划分步骤，不作为创新模块参与开关。论文消融表包含五行，但只需运行四个物理模式：

| 论文行 | 物理运行 | H | P | E | 指标来源 |
|---|---|---:|---:|---:|---|
| Unified Reconstruction Core（URC） | `core` |  |  |  | `tailguard_summary.json -> final_eval_summary` |
| + Dynamic Head Purification | `h_only` | ✓ |  |  | `tailguard_summary.json -> final_eval_summary` |
| + Raw-Tail Enhancement | `h_raw_e` | ✓ |  | ✓ | `memory_eval_summary.json -> metrics_by_mode.fused.summary` |
| TailGuard w/o Enhancement | `full` | ✓ | ✓ |  | `memory_eval_summary.json -> metrics_by_mode.recon.summary` |
| TailGuard | `full` | ✓ | ✓ | ✓ | `memory_eval_summary.json -> metrics_by_mode.fused.summary` |

`full` 的 reconstruction 和 fused 指标来自同一个模型、同一次测试遍历。E 只在训练完成后建立 memory 并融合分数，不参与优化器更新，因此同一次 `full` 运行即可严格配对得到最后两行，不需要额外运行 `full_no_e`。四个模式乘以 MVTec AD/VisA 的 Pareto、Step-K4、Step-K1，共 `4 x 6 = 24` 次训练。

统一输出目录为：

```text
../results/tailguard/dependency_ablation_v5
```

以下按四批编排，每批复用 GPU 1--6。**上一批六条全部结束后再启动下一批**；若并行跨批，请先修改 `--gpus`，避免同卡冲突。服务器开跑前还应确认 `../LTN_visa/visa-{pareto,step_k4,step_k1}-seed01` 已挂载。

### 第一批：URC / `core`（6 次）

`core` 只保留共享 TailSampler 元数据缓存，关闭 H/P/E、GBPS 动态监测、Stage 2、伪类和 memory；缓存与评估过程会恢复随机状态，不应改变连续 reconstruction 训练轨迹。

```bash
# 1. URC / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_pareto_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 1

# 2. URC / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k4_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 2

# 3. URC / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k1_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 3

# 4. URC / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_pareto_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 4

# 5. URC / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k4_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 5

# 6. URC / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k1_seed01_core \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode core \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode none \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.0 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 6
```

### 第二批：H-only / `h_only`（6 次）

`h_only` 只在初始头部候选中执行动态净化；所有初始尾候选均受保护并保留，但不运行 RGD、伪类或 memory。

```bash
# 7. H-only / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_pareto_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 1

# 8. H-only / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k4_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 2

# 9. H-only / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k1_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 3

# 10. H-only / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_pareto_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 4

# 11. H-only / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k4_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 5

# 12. H-only / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k1_seed01_h_only \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_only \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --no-tg_save_cls_embeddings \
  --no-tg_memory_enable \
  --gpus 6
```

### 第三批：H + Raw-Tail Enhancement / `h_raw_e`（6 次）

`h_raw_e` 跳过 P/RGD，将所有初始尾候选直接视为 open-tail，随后建立伪类与条件式 memory，用于隔离 P 的归属校正价值。

```bash
# 13. H + Raw-E / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_pareto_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 14. H + Raw-E / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k4_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 15. H + Raw-E / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k1_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 16. H + Raw-E / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_pareto_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4

# 17. H + Raw-E / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k4_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 5

# 18. H + Raw-E / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k1_seed01_h_raw_e \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode h_raw_e \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 6
```

### 第四批：Full TailGuard / `full`（6 次）

`full` 执行 H、P 和 E。P 对全部初始尾候选只做保护与 RGD 归属校正，不删除任何尾候选；E 只对最终 open-tail 路由融合 memory 分数。

```bash
# 19. Full / MVTec AD / Pareto / seed01 / GPU 1
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-pareto-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_pareto_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 1

# 20. Full / MVTec AD / Step-K4 / seed01 / GPU 2
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k4-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k4_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 2

# 21. Full / MVTec AD / Step-K1 / seed01 / GPU 3
python dinomaly_mvtec_tailguard.py \
  --data_path ../LTN_datasets/mvtecad-step_k1-seed01 \
  --diag_manifest_path ./manifest/mvtecad-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name mvtec_step_k1_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 3

# 22. Full / VisA / Pareto / seed01 / GPU 4
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-pareto-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/pareto/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_pareto_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 4

# 23. Full / VisA / Step-K4 / seed01 / GPU 5
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k4-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k4/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k4_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 5

# 24. Full / VisA / Step-K1 / seed01 / GPU 6
python dinomaly_visa_tailguard.py \
  --data_path ../LTN_visa/visa-step_k1-seed01 \
  --diag_manifest_path ./manifest/visa-nlt/step_k1/seed01/inject_defects.txt \
  --save_dir ../results/tailguard/dependency_ablation_v5 \
  --save_name visa_step_k1_seed01_full \
  --total_iters 10000 \
  --eval_interval 5000 \
  --tg_method_mode full \
  --tailsampler_type adaptive_trim_mode \
  --tg_tail_embedding_source encoder_cls \
  --tg_group_embedding_source encoder \
  --gbps_auto_k \
  --gbps_k_candidates 6,8,10,12,15,20 \
  --tg_stage1_training_scope all_samples \
  --gbps_postprocess_mode remove \
  --gbps_prune_mode adaptive \
  --gbps_prune_max_ratio 0.10 \
  --gbps_prune_stable_window 3 \
  --gbps_prune_stable_min_observations 1 \
  --gbps_prune_min_active_ratio 0.50 \
  --gbps_min_keep_per_group 20 \
  --tg_attachment_membership_mode rgd \
  --tg_rgd_split_mode segmented_bic \
  --tg_stable_window 3 \
  --tg_stable_min_observations 1 \
  --tg_t_attached_noise_p_high_thr 1.0 \
  --tg_save_cls_embeddings \
  --tg_memory_enable \
  --tg_memory_fusion_lambda 1.0 \
  --tg_memory_topk_ratio 0.05 \
  --gpus 6
```

### 完成验收与消融表取值

每条运行都先检查：

```text
<run>/final_model.pt                    -> iteration == 10000
<run>/tailguard/tailguard_summary.json  -> schema_version == 5，method_mode 与 save_name 一致
```

再按模式核验：

| 模式 | 必须满足 |
|---|---|
| `core` | `memory_status == "disabled"`；`stage2_summary == null`；`gbps_trigger_summary == null` |
| `h_only` | `memory_status == "disabled"`；`stage2_summary.reconciliation_performed == false`；`stage2_summary.enhancement_performed == false`；`stage2_summary.num_tail_candidates_removed == 0` |
| `h_raw_e` | `memory_status == "completed"`；`stage2_summary.reconciliation_performed == false`；`stage2_summary.num_tail_open == prepare_summary.num_tail_candidates`；`stage2_summary.num_tail_candidates_removed == 0` |
| `full` | `memory_status == "completed"`；`stage2_summary.reconciliation_performed == true`；`stage2_summary.enhancement_performed == true`；`stage2_summary.num_tail_candidates_removed == 0` |

若 `h_only`、`h_raw_e` 或 `full` 的 `h_zero_removal_fallback == true`，表示 H 在该场景证据不足并安全退化为零删除；P/E 仍应继续完成，这不是程序失败，但拉表时必须一并记录。

六个场景均按以下规则取论文消融指标：

```text
URC                  -> <core>/tailguard/tailguard_summary.json
                        final_eval_summary
H-only               -> <h_only>/tailguard/tailguard_summary.json
                        final_eval_summary
H + Raw-E             -> <h_raw_e>/tailguard/memory/memory_eval_summary.json
                        metrics_by_mode.fused.summary
H + P (w/o E)         -> <full>/tailguard/memory/memory_eval_summary.json
                        metrics_by_mode.recon.summary
Full H + P + E        -> <full>/tailguard/memory/memory_eval_summary.json
                        metrics_by_mode.fused.summary
```

P 的性能贡献比较 `Full fused - H + Raw-E fused`；E 的性能贡献比较同一个 Full 文件中的 `fused - recon`。不要把 H+P reconstruction 与 H-only reconstruction 的差异解释为 P 的贡献，因为 P 只校正尾候选角色，不改变 reconstruction 训练集。

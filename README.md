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
```
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

```bash
python dinomaly_mvtec_uni.py --data_path ../LTN_datasets/mvtecad-pareto-seed02 --save_dir ../saved_results/phase2/ --save_name mvtec --gpus 1 --warmup_auto_end --diag_manifest_path ./manifest/mvtecad-nlt/pareto/seed02/inject_defects.txt
```

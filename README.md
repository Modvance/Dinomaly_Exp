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
```
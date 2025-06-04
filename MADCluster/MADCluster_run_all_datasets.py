import re
import subprocess
import argparse

DATASET_INFO = {
    'MSL': ['C-1', 'C-2', 'D-14', 'D-15', 'D-16', 'F-4', 'F-5', 'F-7', 'F-8', 'M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-10', 'P-11', 'P-14', 'P-15', 'S-2', 'T-4', 'T-5', 'T-8', 'T-9', 'T-12', 'T-13'],
    'SMAP': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9', 'B-1', 'D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-11', 'D-12', 'D-13', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'F-1', 'F-2', 'F-3', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'P-1', 'P-2', 'P-3', 'P-4', 'P-7', 'R-1', 'S-1', 'T-1', 'T-2', 'T-3'],
    'SMD': ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-1-4.txt', 'machine-1-5.txt', 'machine-1-6.txt', 'machine-1-7.txt', 'machine-1-8.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt', 'machine-2-4.txt', 'machine-2-5.txt', 'machine-2-6.txt', 'machine-2-7.txt', 'machine-2-8.txt', 'machine-2-9.txt', 'machine-3-1.txt', 'machine-3-2.txt', 'machine-3-3.txt', 'machine-3-4.txt', 'machine-3-5.txt', 'machine-3-6.txt', 'machine-3-7.txt', 'machine-3-8.txt', 'machine-3-9.txt', 'machine-3-10.txt', 'machine-3-11.txt'],
    'PSM': []
}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def run(dataset, objective, madcluster):
    dataset_names = DATASET_INFO[dataset]

    madcluster_flag = "--MADCluster" if madcluster else ""

    if dataset == 'PSM':
        command = f"""python3 MADCluster_MAIN.py \
                {madcluster_flag} \
                --batch_size 128 \
                --window_size 100 \
                --d_model 256 \
                --patch_size 2 5 10 \
                --n_heads 1 \
                --e_layers 3 \
                --k 3 \
                --temperature 50 \
                --nu 0.01 \
                --objective {objective} \
                \
                \
                --smoothing_factor 0.01 \
                --lr 1e-4 \
                --lr_lambda 0.9 \
                --min_delta 1e-5 \
                --weight_decay 0.00001 \
                --init_method he \
                --init_threshold 0.5 \
                \
                \
                --dataset PSM \
                --dataset_path ../datasets/ \
                --num_epochs 30 \
                --patience 5 \
                --device '0'"""
        subprocess.run(command, shell=True)

    else:
        dataset_names_sorted = sorted(dataset_names, key=natural_sort_key)

        for name in dataset_names_sorted:
            command = f"""python3 MADCluster_MAIN.py \
                {madcluster_flag} \
                --batch_size 128 \
                --window_size 100 \
                --d_model 256 \
                --patch_size 2 5 10 \
                --n_heads 1 \
                --e_layers 3 \
                --k 3 \
                --temperature 50 \
                --nu 0.01 \
                --objective {objective} \
                \
                \
                --smoothing_factor 0.01 \
                --lr 1e-4 \
                --lr_lambda 0.9 \
                --min_delta 1e-5 \
                --weight_decay 0.00001 \
                --init_method he \
                --init_threshold 0.5 \
                \
                \
                --dataset {dataset} \
                --dataset_path ../datasets/ \
                --fname {name} \
                --num_epochs 30 \
                --patience 5 \
                --device '0'"""

            subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["one-class", "soft-boundary"], required=True)
    parser.add_argument("--MADCluster", action="store_true")
    args = parser.parse_args()

    for dataset in DATASET_INFO.keys():
        print(f"\n========== Running {dataset} ({args.objective}) ==========")
        run(dataset, args.objective, args.MADCluster)

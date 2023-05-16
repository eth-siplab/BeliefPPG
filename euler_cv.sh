DATA="/cluster/scratch/${USER}/ppg"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy

pip install -q -r requirements.txt

WRAP="python train_eval.py --data_dir ${DATA} --dataset dalia --use_wandb"
echo ${WRAP}
sbatch \
    --time=05:00:00 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=20 \
    -J "beauppg-split-${SPLIT}" \
    --mem-per-cpu=5000 \
    --gres=gpumem:10240m \
    --gpus=1 \
    --mail-type=ALL \
    --mail-user="${USER}@ethz.ch" \
    --output="Output/log.txt" \
    --wrap="${WRAP}"

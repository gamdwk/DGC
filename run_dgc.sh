docker exec -it dgc1 bash
conda activate GCond
service ssh start
python3 /home/sj/nvme/DGC/launch.py \
--workspace /home/sj/nvme/DGC \
--num_trainers 1 \
--num_samplers 1 \
--num_servers 1 \
--extra_envs PYTHONPATH=/home/sj/sdb/anaconda3/envs/dgs/bin/python3:$PYTHONPATH \
--part_config data/ogbn-arxiv.json \
--ip_config ip_config.txt \
"/home/sj/sdb/anaconda3/envs/dgs/bin/python3 train_dgc_tranductive.py --ip_config ip_config.txt --epochs 30 --num_gpus 2"

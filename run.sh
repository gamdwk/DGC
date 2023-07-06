cd /home/sj/nvme/DGC
python3 /home/sj/nvme/DGC/launch.py \
--workspace /home/sj/nvme/DGC \
--num_trainers 1 \
--num_samplers 1 \
--num_servers 1 \
--extra_envs PYTHONPATH=/home/sj/sdb/anaconda3/envs/dgs/bin/python3:$PYTHONPATH \
--part_config data/ogbn-arxiv.json \
--ip_config ip_config.txt \
"/home/sj/sdb/anaconda3/envs/dgs/bin/python3 train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 100 --batch_size 1024 --num_gpus 2"

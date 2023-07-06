cd /home/sj/nvme/DGC
python3 partition_graph.py --dataset ogbn-arxiv --num_parts 2
python3 partition_graph.py --dataset ogb-product --num_parts 2
python3 partition_graph.py --dataset amazon --num_parts 2
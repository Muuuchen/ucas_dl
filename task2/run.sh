#!/bin/bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py
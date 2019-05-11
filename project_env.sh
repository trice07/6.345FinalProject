#!/bin/bash

module load engaging/python/2.7.10
module load engaging/python/3.5.0
module load gcc/4.8.4
module load cuda/8.0
module load engaging/OpenBlas/0.2.14

module list

. torch_env/bin/activate

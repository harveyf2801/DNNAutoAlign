import os

# Get the world size from the WORLD_SIZE variable or directly from SLURM:
world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
# Likewise for RANK and LOCAL_RANK:
rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

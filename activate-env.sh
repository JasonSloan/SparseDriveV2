#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请在当前终端执行：source ${BASH_SOURCE[0]}"
    exit 1
fi

source /root/software/miniconda3/etc/profile.d/conda.sh

export NAVSIM_DEVKIT_ROOT=/mnt/nfs_docker_volume/training-container-space/code/end2end/SparseDriveV2
export OPENSCENE_DATA_ROOT=/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/openscene
export NUPLAN_MAPS_ROOT=/mnt/nfs_docker_volume/training-container-space/mnt/datasets/object-detection-datasets/open-source/nuPlan/maps
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
mkdir -p "$NAVSIM_EXP_ROOT"

conda activate sparsedrive
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

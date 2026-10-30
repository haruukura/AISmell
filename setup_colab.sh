#!/usr/bin/env bash
# setup_colab.sh — Run this as the FIRST cell in Google Colab (T4 GPU runtime).
# Order matters: torch → dgl → rdkit → deepchem → openpom → iterstrat
set -euo pipefail

echo "=== [1/6] Installing PyTorch 2.4.0 + CUDA 12.1 ==="
pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== [2/6] Installing DGL (CUDA 12.1) ==="
pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

echo "=== [3/6] Installing RDKit ==="
pip install -q rdkit-pypi

echo "=== [4/6] Installing DeepChem (pre-release) ==="
pip install -q --pre deepchem

echo "=== [5/6] Installing OpenPOM ==="
pip install -q openpom

echo "=== [6/6] Installing remaining dependencies ==="
pip install -q iterative-stratification tqdm seaborn

python -c "
import torch, dgl, deepchem, rdkit, openpom
print(f'torch      {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'dgl        {dgl.__version__}')
print(f'deepchem   {deepchem.__version__}')
print(f'rdkit      {rdkit.__version__}')
print(f'openpom    {openpom.__version__}')
print('All dependencies installed successfully.')
"

# CDBS-Net

---
## 🛠️ Environment Initialization & Setup 

Execute the following command block in your terminal to initialize the environment, deploy PyTorch, and install all required dependencies at once:

### 1. Create and Activate the Isolated Conda Sub-Environment
```bash
conda create -n cdbsnet python=3.9 -y
conda activate cdbsnet
```
### 2. Deploy PyTorch ecosystem aligned with CUDA 11.7
```bash
pip install torch==2.5.1+cu117 torchvision==0.20.1+cu117 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
```
### 3. Install essential core downstream dependencies
```bash
pip install opencv-python imageio numpy tabulate
```

## 📂 Standardized Data Topology
Extract your downloaded dataset packages and map them to the structural layout below inside the ./dataset/ root pathway:

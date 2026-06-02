# CDBS-Net

---

## 🛠️ Environment Initialization & Sanity Check

We recommend utilizing Anaconda to build an isolated runtime sub-environment. Run the following command stream to guarantee clean dependency separation:

```bash
# 1. Initialize virtual sub-environment
conda create -n cdbsnet python=3.10 -y
conda activate cdbsnet

# 2. Deploy PyTorch ecosystem aligned with CUDA 11.7
pip install torch==2.5.1+cu117 torchvision==0.20.1+cu117 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

# 3. Deploy utility dependencies
pip install opencv-python imageio numpy tabulate

📂 Standardized Data Topology
Extract your downloaded dataset packages and map them to the structural layout below inside the ./dataset/ root pathway:

# CDBS-Net

---

## 🛠️ Environment Setup & Installation

To eliminate dependency conflicts and ensure smooth reproduction, we recommend managing your runtime framework via Anaconda. Execute the following commands sequentially to initialize the self-contained environment matching our hardware configuration (tested on a single NVIDIA GPU):

### 1. Create and activate the isolated conda sub-environment
```bash
conda create -n cdbsnet python=3.9 -y
conda activate cdbsnet

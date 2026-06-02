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
pip install torch==2.5.1+cu117 torchvision==0.20.1+cu117
```
### 3. Install essential core downstream dependencies
```bash
pip install opencv-python imageio numpy tabulate
```

## 📂 Standardized Data Topology
Extract your downloaded dataset packages and map them to the structural layout below inside the ./dataset/ root pathway:
```bash
dataset/
├── TrainDataset/                
│   ├── CVC-ClinicDB/            
│   │   ├── images/
│   │   └── masks/
│   └── Kvasir-SEG/            
│       ├── images/
│       └── masks/
└── TestDataset/                
    ├── CVC-300/             
    ├── CVC-ClinicDB/          
    ├── CVC-ColonDB/            
    ├── ETIS-LaribPolypDB/      
    └── Kvasir/                  
        ├── images/
        └── masks/
```
## 🚀 Execution Pipelines

### 1. Training Phase
The training loop runs under a polynomial decay scheduling layer with an initial learning rate of 0.001, optimizing the parameters for 80 epochs via Stochastic Gradient Descent (SGD).

Launch standard network optimization flow:
```bash
python Train.py
```

### 2. Inference & Saliency Map Generation
The Test.py execution layer acts as a unified automated loop. It sweeps across all testing sub-directories inside ./dataset/TestDataset/, loads the network weights, and exports binary predictions.
```bash
python Test.py
```

### 3. Quantitative Performance Metrics Benchmarking
The metric scoring pipeline runs via predict_score.py, evaluating the masks over 256 structural sliding thresholds to produce exact medical evaluation indicators.
```bash
python predict_score.py
```

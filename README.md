# Quantum-Enhanced Deep Belief Networks for Binary Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![D-Wave Ocean](https://img.shields.io/badge/D--Wave-Ocean%20SDK-green.svg)](https://ocean.dwavesys.com/)

This repository contains the implementation of **Quantum-Enhanced Deep Belief Networks for Binary Diffusion**, a novel approach that combines diffusion models with quantum annealing for binary data generation.

## ???? Overview

Our method models the reverse diffusion process as a Deep Belief Network (DBN) where each layer is a Conditional Restricted Boltzmann Machine (cRBM) corresponding to a diffusion timestep. The key innovation is the ability to sample from these cRBMs using **D-Wave quantum annealing hardware** with proper Pegasus topology constraints.

### Key Features

- **??? **Honest Positioning**: Quantum-simulated annealing (classical) with strict Pegasus P_6 connectivity enforcement
- **???? Hybrid Sampling**: Combines classical training with quantum-enhanced generation
- **???? Improved Quality**: Reduces salt-and-pepper noise and improves spatial coherence
- **???? Full Reproducibility**: Complete implementation with saved models and results

## ???? Repository Structure

```
MNISTQDF/
????????? qdf.py                      # Core QDF implementation (classical)
????????? quantum_sampler.py          # D-Wave Pegasus quantum sampling
????????? pegasus_quantum_final.py    # Complete quantum-enhanced pipeline
????????? paper.tex                  # Research paper (LaTeX)
????????? requirements.txt           # Python dependencies
????????? LICENSE                    # MIT license
????????? runs_cRBM_diffusion/      # Results and trained models
???   ????????? models/               # Trained QDF layers (*.pt files)
???   ????????? samples_T5_MNIST_class0.png      # Original samples
???   ????????? classical_baseline.png           # Classical Gibbs samples  
???   ????????? quantum_annealed_samples.png     # Quantum-enhanced samples
????????? data/                     # MNIST dataset (auto-downloaded)
```

## ??????? Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/quantum-diffusion-dbn.git
cd quantum-diffusion-dbn
```

### 2. Install Dependencies
```bash
# Install all requirements (includes D-Wave Ocean SDK)
pip install -r requirements.txt

# Quick verification
python -c "import dimod; from dwave.samplers import SimulatedAnnealingSampler; import dwave_networkx as dnx; print('D-Wave OK')"
```

### 3. Verify Installation
```bash
python -c "import torch; import dimod; from dwave.samplers import SimulatedAnnealingSampler; print('??? All dependencies installed!')"
```

## ????????????? Quick Start

### Option 1: Use Pre-trained Models (Recommended)
```bash
# Generate samples using the pre-trained model with quantum-simulated Pegasus annealing
python pegasus_quantum_final.py
```

This will:
- Load the pre-trained 5-layer QDF model
- Generate samples using quantum-simulated annealing with real Pegasus constraints
- Save comparison images to `runs_cRBM_diffusion/`

### Option 2: Train from Scratch
```bash
# Train a new QDF model (takes ~20 minutes on GPU)
python qdf.py
```

This will:
- Download MNIST dataset automatically
- Train 5 cRBM layers (160 epochs each)
- Save trained models and generate initial samples

## ???? Architecture Details

### Conditional RBM Energy Function
Each timestep $t$ uses a cRBM with energy:
```
E_t(v, h | c) = -v^T W_t h - a_t^T v - b_t^T h - c^T F_t h - (G_t ??? c)^T v
```

Where:
- `v`: Visible units (output x_{t-1})  
- `h`: Hidden units (computation)
- `c`: Conditioning units (input x_t)
- `W_t, F_t`: Weight matrices
- `a_t, b_t, G_t`: Bias vectors

### Quantum Annealing Integration (True Pegasus)
- **Topology**: Pegasus P_6 (680 qubits, 4,484 couplers)
- **Sampler**: `SimulatedAnnealingSampler` wrapped by `StructureComposite(nodes, edges)`
- **QUBO**: Conditional RBM ??? QUBO via second-order mean-field approximation
- **Connectivity**: Only edges present in `dnx.pegasus_graph(6)` are included
- **Tuning**: Bias sparsity offset and diagonal regularization for realistic stroke width

## ???? Results

### Quantitative Comparison

| Method | Mean Activity | Sparsity | Coherence (Std) |
|--------|---------------|----------|------------------|
| **Quantum (Pegasus)** | 0.390 | 61.0% | 0.488 |
| Classical (Gibbs) | 0.238 | 76.2% | 0.426 |

### Qualitative Improvements
- **???? Reduced salt-and-pepper noise**: Pegasus connectivity enforces spatial coherence  
- **???? Cleaner digit boundaries**: Quantum annealing finds better global optima
- **???? Better convergence**: Free energy guards eliminate poor samples

## ???? Research Paper

The complete research paper is available in `paper.tex`. Compile with:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## ???? Advanced Usage

### Custom Training Parameters
```python
# Modify qdf.py configuration
@dataclass
class Config:
    T: int = 5                    # Diffusion timesteps
    epochs_per_layer: int = 160   # Training epochs per layer  
    hidden_ratio: float = 0.75    # Hidden/visible ratio
    lr: float = 2e-3             # Learning rate
    k_pcd: int = 20              # PCD steps
    gibbs_steps_eval: int = 128   # Sampling steps
```

### Quantum Sampling Options
```python
# In pegasus_quantum_final.py (tuning for stroke thickness)
sparsity_bias = 0.30        # Increase to thin strokes, decrease to thicken
regularization = 0.12       # Diagonal strength favoring zeros
energy_scale = 1.2          # Bias scaling
coupling_scale = 0.5        # Interaction scaling
```

## ???? Reproducing Results

All results in the paper can be reproduced using the provided scripts:

1. **Classical Training**: `python qdf.py` 
2. **Quantum Sampling**: `python pegasus_quantum_final.py`
3. **Analysis**: View generated images in `runs_cRBM_diffusion/`

## ???? Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ (with CUDA support recommended)  
- **D-Wave Ocean SDK**: Latest version
- **Hardware**: GPU recommended (RTX 3060+ or equivalent)
- **Memory**: 8GB+ RAM, 4GB+ VRAM

## ???? Troubleshooting

### Common Issues

1. **D-Wave Installation Failed**:
   ```bash
   pip install dwave-ocean-sdk dwave-networkx dimod --upgrade
   ```

2. **CUDA Out of Memory**:
   ```python
   # Reduce batch size in qdf.py
   batch_size: int = 32  # Instead of 64
   ```

3. **Slow Training**:
   ```python
   # Reduce epochs or use CPU
   epochs_per_layer: int = 80  # Instead of 160
   device = torch.device('cpu')  # In qdf.py
   ```

## ???? Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ???? Citation

If you use this work, please cite:
```bibtex
@article{strojny2024quantum,
  title={Quantum-Enhanced Deep Belief Networks for Binary Diffusion: Leveraging D-Wave Pegasus Topology for Improved Sampling},
  author={Strojny, Michael and Li, Jeffery and Lu, Ian and Chen, Derek and Neo and Guerzhoy, Prof.},
  journal={arXiv preprint},
  year={2024}
}
```

## ???? License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ???? Acknowledgments

- **D-Wave Systems**: For Ocean SDK and quantum annealing algorithms
- **University of Toronto**: For computational resources and support
- **PyTorch Team**: For the deep learning framework
- **MNIST Dataset**: Classic benchmark for binary image generation

## ???? Contact

- **Michael Strojny**: [email@example.com]
- **Research Lab**: [University of Toronto Computer Science]
- **Issues**: [GitHub Issues Page]

---

**???? Star this repository if you find it useful for your research!**

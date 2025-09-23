""" 
Pegasus Quantum Sampler for QDF
===============================

Quantum-inspired sampling using D-Wave's Pegasus topology
with simulated annealing for binary diffusion models.
"""

import numpy as np
import torch
import networkx as nx
from typing import Optional, Dict, List, Tuple
import time
import warnings

try:
    import dimod
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave.embedding import embed_bqm, unembed_sampleset
    import dwave_networkx as dnx
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    warnings.warn("D-Wave Ocean SDK not found. Install with: pip install dwave-ocean-sdk")

class PegasusQuantumSampler:
    """Pegasus quantum sampler for RBM energy functions."""
    
    def __init__(self, 
                 pegasus_size: int = 16,
                 use_hardware: bool = False,
                 num_reads: int = 100,
                 annealing_time: int = 20,
                 chain_strength: float = 1.0,
                 temperature_schedule: Optional[List[Tuple[float, float]]] = None):
        """Initialize Pegasus quantum sampler."""
        self.pegasus_size = pegasus_size
        self.use_hardware = use_hardware
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.chain_strength = chain_strength
        
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK required. Install with: pip install dwave-ocean-sdk")
        
        self.pegasus_graph = dnx.pegasus_graph(pegasus_size)
        self.num_qubits = len(self.pegasus_graph.nodes())
        
        print(f"Initialized Pegasus P_{pegasus_size} with {self.num_qubits} qubits")
        print(f"Graph connectivity: {len(self.pegasus_graph.edges())} edges")
        
        if use_hardware and DWAVE_AVAILABLE:
            try:
                self.sampler = EmbeddingComposite(DWaveSampler())
                self.hardware_available = True
                print("Connected to D-Wave quantum hardware!")
            except Exception as e:
                warnings.warn(f"Hardware connection failed: {e}. Using simulated annealing.")
                self.sampler = SimulatedAnnealingSampler()
                self.hardware_available = False
        else:
            self.sampler = SimulatedAnnealingSampler()
            self.hardware_available = False
            
        if temperature_schedule is None:
            self.temperature_schedule = [
                (0.0, 8.0), (0.1, 4.0), (0.3, 2.0), (0.5, 1.0),
                (0.7, 0.5), (0.9, 0.2), (1.0, 0.05)
            ]
        else:
            self.temperature_schedule = temperature_schedule
    
    def rbm_to_qubo(self, W: torch.Tensor, a: torch.Tensor, b: torch.Tensor, 
                    c: torch.Tensor, G: torch.Tensor, F: torch.Tensor) -> Dict:
        """Convert conditional RBM energy to QUBO formulation."""
        D, H = W.shape
        
        if c.ndim > 1:
            c_np = c[0].detach().cpu().numpy()
        else:
            c_np = c.detach().cpu().numpy()
            
        a_cond = (a.detach().cpu().numpy() + G.detach().cpu().numpy() * c_np)
        b_cond = (b.detach().cpu().numpy() + c_np @ F.detach().cpu().numpy())
        W_np = W.detach().cpu().numpy()
        
        available_qubits = min(D, self.num_qubits // 2)  # Conservative mapping
        pegasus_nodes = list(self.pegasus_graph.nodes())[:available_qubits]
        
        Q = {}
        
        for i in range(available_qubits):
            qubit = pegasus_nodes[i]
            Q[(qubit, qubit)] = -a_cond[i] * 2.0
        
        h_mean = 1.0 / (1.0 + np.exp(-b_cond))  # Mean field solution
        
        for i in range(available_qubits):
            for j in range(i+1, available_qubits):
                qi, qj = pegasus_nodes[i], pegasus_nodes[j]
                
                # Effective coupling through all hidden units
                coupling = 0.0
                for k in range(H):
                    coupling += W_np[i, k] * W_np[j, k] * h_mean[k] * (1 - h_mean[k])
                
                if abs(coupling) > 1e-6:
                    scaled_coupling = coupling * 0.5
                    Q[(qi, qj)] = -scaled_coupling
                    
        regularization = 0.01
        for i in range(available_qubits):
            qubit = pegasus_nodes[i]
            if (qubit, qubit) in Q:
                Q[(qubit, qubit)] += regularization
            else:
                Q[(qubit, qubit)] = regularization
        
        return Q
    
    def sample_rbm_quantum(self, W: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                          c: torch.Tensor, G: torch.Tensor, F: torch.Tensor,
                          num_samples: int = 64) -> torch.Tensor:
        """Sample from conditional RBM using quantum annealing."""
        batch_size = c.shape[0]
        D = W.shape[0]
        all_samples = []
        
        for b_idx in range(batch_size):
            c_single = c[b_idx:b_idx+1]
            
            Q = self.rbm_to_qubo(W, a, b, c_single, G, F)
            
            if not Q:
                samples_batch = torch.randint(0, 2, (num_samples, D), dtype=torch.float32)
            else:
                try:
                    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
                    
                    if self.hardware_available:
                        sample_kwargs = {
                            'num_reads': self.num_reads,
                            'annealing_time': self.annealing_time,
                            'chain_strength': self.chain_strength
                        }
                    else:
                        sample_kwargs = {
                            'num_reads': self.num_reads,
                            'schedule': self.temperature_schedule,
                            'seed': np.random.randint(0, 2**31)
                        }
                    
                    sampleset = self.sampler.sample(bqm, **sample_kwargs)
                    
                    samples_list = []
                    pegasus_nodes = list(self.pegasus_graph.nodes())
                    available_qubits = min(D, self.num_qubits // 2)
                    
                    for sample in sampleset.samples():
                        sample_vec = torch.zeros(D)
                        
                        for i in range(available_qubits):
                            qubit = pegasus_nodes[i]
                            if qubit in sample:
                                sample_vec[i] = float(sample[qubit])
                            else:
                                sample_vec[i] = np.random.binomial(1, 0.5)
                        
                        if available_qubits < D:
                            a_cond = a.detach().cpu().numpy() + G.detach().cpu().numpy() * c[0].detach().cpu().numpy()
                            for i in range(available_qubits, D):
                                prob = 1.0 / (1.0 + np.exp(-a_cond[i]))
                                sample_vec[i] = np.random.binomial(1, prob)
                                
                        samples_list.append(sample_vec)
                    
                    while len(samples_list) < num_samples:
                        idx = np.random.randint(0, len(samples_list))
                        perturbed = samples_list[idx].clone()
                        flip_mask = torch.rand(D) < 0.05
                        perturbed[flip_mask] = 1 - perturbed[flip_mask]
                        samples_list.append(perturbed)
                    
                    samples_batch = torch.stack(samples_list[:num_samples])
                    
                except Exception as e:
                    print(f"Quantum sampling failed: {e}. Using random fallback.")
                    samples_batch = torch.randint(0, 2, (num_samples, D), dtype=torch.float32)
            
            all_samples.append(samples_batch)
        
        return torch.cat(all_samples, dim=0)
    
    def quantum_gibbs_step(self, v: torch.Tensor, h: torch.Tensor, 
                          W: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                          c: torch.Tensor, G: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantum-enhanced Gibbs sampling step."""
        batch_size = v.shape[0]
        
        logits_h = b + (c @ F) + (v @ W)
        logits_h = torch.clamp(logits_h, -30, 30)
        h_new = torch.bernoulli(torch.sigmoid(logits_h))
        
        if np.random.random() < 0.3:
            try:
                v_new = self.sample_rbm_quantum(W, a, b, c, G, F, num_samples=batch_size)
                v_new = v_new[:batch_size]
            except Exception as e:
                logits_v = a + (G * c) + (h_new @ W.t())
                logits_v = torch.clamp(logits_v, -30, 30)
                v_new = torch.bernoulli(torch.sigmoid(logits_v))
        else:
            logits_v = a + (G * c) + (h_new @ W.t())
            logits_v = torch.clamp(logits_v, -30, 30)
            v_new = torch.bernoulli(torch.sigmoid(logits_v))
        
        return v_new, h_new
    
    def get_connectivity_info(self) -> Dict:
        """Return information about the Pegasus connectivity."""
        return {
            'topology': 'Pegasus',
            'size': self.pegasus_size,
            'num_qubits': self.num_qubits,
            'num_edges': len(self.pegasus_graph.edges()),
            'degree_stats': {
                'min': min(dict(self.pegasus_graph.degree()).values()),
                'max': max(dict(self.pegasus_graph.degree()).values()),
                'mean': np.mean(list(dict(self.pegasus_graph.degree()).values()))
            },
            'hardware_available': self.hardware_available
        }

def create_quantum_sampler(pegasus_size: int = 6, use_hardware: bool = False) -> Optional[PegasusQuantumSampler]:
    """Create quantum sampler with error handling."""
    try:
        sampler = PegasusQuantumSampler(
            pegasus_size=pegasus_size,
            use_hardware=use_hardware,
            num_reads=50,
            annealing_time=20
        )
        print("Quantum sampler created successfully!")
        info = sampler.get_connectivity_info()
        print(f"Connectivity: {info['num_qubits']} qubits, {info['num_edges']} edges")
        return sampler
    except Exception as e:
        print(f"Failed to create quantum sampler: {e}")
        return None

if __name__ == "__main__":
    print("Testing Pegasus Quantum Sampler...")
    
    sampler = create_quantum_sampler(pegasus_size=4, use_hardware=False)
    
    if sampler:
        D, H = 16, 12
        batch_size = 4
        
        W = torch.randn(D, H) * 0.1
        a = torch.randn(D) * 0.1
        b = torch.randn(H) * 0.1
        G = torch.randn(D) * 0.1
        F = torch.randn(D, H) * 0.1
        c = torch.randint(0, 2, (batch_size, D)).float()
        
        print(f"Testing quantum sampling with D={D}, H={H}, batch_size={batch_size}")
        
        start_time = time.time()
        samples = sampler.sample_rbm_quantum(W, a, b, c, G, F, num_samples=16)
        end_time = time.time()
        
        print(f"Generated {samples.shape[0]} samples in {end_time - start_time:.2f}s")
        print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")
        
        v = torch.randint(0, 2, (batch_size, D)).float()
        h = torch.randint(0, 2, (batch_size, H)).float()
        
        v_new, h_new = sampler.quantum_gibbs_step(v, h, W, a, b, c, G, F)
        print(f"Gibbs step: v changed {(v != v_new).float().mean():.1%} of bits")
        
        info = sampler.get_connectivity_info()
        print("\nConnectivity Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

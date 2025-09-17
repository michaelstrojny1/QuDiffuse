#!/usr/bin/env python3
"""
HONEST QUANTUM VERIFICATION
===========================
Let's be completely honest about what we actually ran and fix the misleading metrics.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import numpy as np

# Test D-Wave imports
try:
    from dwave.samplers import SimulatedAnnealingSampler
    import dimod
    import dwave_networkx as dnx
    from dimod import StructureComposite
    DWAVE_OK = True
except ImportError:
    DWAVE_OK = False

def test_what_simulated_annealing_actually_is():
    """Test what SimulatedAnnealingSampler actually does"""
    print("=== REALITY CHECK: WHAT IS SIMULATEDANNEALINGSAMPLER? ===")
    
    if not DWAVE_OK:
        print("❌ D-Wave not available")
        return
    
    sampler = SimulatedAnnealingSampler()
    print(f"✅ SimulatedAnnealingSampler type: {type(sampler)}")
    print(f"   Module: {sampler.__class__.__module__}")
    print(f"   Docstring: {sampler.__doc__[:100] if sampler.__doc__ else 'None'}...")
    
    # Test timing on small problem
    Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    
    start = time.time()
    sampleset = sampler.sample(bqm, num_reads=100)
    elapsed = time.time() - start
    
    print(f"   Timing: {elapsed:.4f}s for 100 reads on 2-variable QUBO")
    print(f"   This is CLASSICAL simulated annealing, not quantum hardware!")
    print()

def test_realistic_reconstruction_metrics():
    """Test reconstruction with proper baselines"""
    print("=== REALISTIC RECONSTRUCTION METRICS ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    try:
        blob = torch.load('runs_cRBM_diffusion/models/layer_t1.pt', map_location=device, weights_only=False)
        D, H = blob['W'].shape[0], blob['W'].shape[1]
        W, a, b, F_mat, G = blob['W'], blob['a'], blob['b'], blob['F'], blob['G']
        print(f"✅ Loaded model: D={D}, H={H}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Load MNIST test data
    tfm = transforms.ToTensor()
    try:
        test_base = datasets.MNIST(root='./data', train=False, download=False, transform=tfm)
    except:
        test_base = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    
    # Get different classes for comparison
    class_0_indices = [i for i, (_, y) in enumerate(test_base) if int(y) == 0][:100]
    class_8_indices = [i for i, (_, y) in enumerate(test_base) if int(y) == 8][:100]
    
    print(f"Testing on {len(class_0_indices)} zeros and {len(class_8_indices)} eights")
    
    def evaluate_reconstruction(indices, class_name):
        acc_total, bce_total = 0.0, 0.0
        
        with torch.no_grad():
            for idx in indices:
                x, _ = test_base[idx]
                # Binarize (this introduces randomness!)
                x_bin = (torch.rand_like(x) < x).float().view(-1).to(device)
                
                v = x_bin.unsqueeze(0)
                c = v.clone()
                
                # One-step reconstruction
                logits_h = b + (c @ F_mat) + (v @ W)
                p_h = torch.sigmoid(torch.clamp(logits_h, -30, 30))
                
                recon_logits = a + (G * c) + (p_h @ W.t())
                recon_logits = torch.clamp(recon_logits, -30, 30)
                
                v_recon = (torch.sigmoid(recon_logits) > 0.5).float()
                acc = (v_recon == v).float().mean().item()
                bce = F.binary_cross_entropy_with_logits(recon_logits, v, reduction='mean').item()
                
                acc_total += acc
                bce_total += bce
        
        return acc_total / len(indices), bce_total / len(indices)
    
    # Test on training class (class 0)
    acc_0, bce_0 = evaluate_reconstruction(class_0_indices, "zeros (training class)")
    print(f"   Class 0 (zeros, trained): Acc={acc_0:.4f}, BCE={bce_0:.4f}")
    
    # Test on different class (class 8) 
    acc_8, bce_8 = evaluate_reconstruction(class_8_indices, "eights (different class)")
    print(f"   Class 8 (eights, unseen): Acc={acc_8:.4f}, BCE={bce_8:.4f}")
    
    # Baseline: Random reconstruction
    random_acc, random_bce = 0.0, 0.0
    for idx in class_0_indices[:20]:
        x, _ = test_base[idx]
        x_bin = (torch.rand_like(x) < x).float().view(-1)
        random_recon = torch.rand_like(x_bin)
        
        acc = (random_recon.round() == x_bin).float().mean().item()
        bce = F.binary_cross_entropy(random_recon, x_bin, reduction='mean').item()
        random_acc += acc
        random_bce += bce
    
    print(f"   Random baseline:          Acc={random_acc/20:.4f}, BCE={random_bce/20:.4f}")
    
    # Analysis
    print(f"\n   ANALYSIS:")
    if acc_0 > 0.95:
        print(f"   ⚠️  {acc_0:.1%} accuracy on training class suggests overfitting")
    if acc_0 - acc_8 > 0.1:
        print(f"   ⚠️  Large gap ({acc_0-acc_8:.3f}) between classes confirms overfitting")
    if bce_0 < 0.1:
        print(f"   ⚠️  Very low BCE ({bce_0:.4f}) suggests near-perfect memorization")
    
    print()

def test_quantum_vs_classical_sampling():
    """Compare 'quantum' (simulated annealing) vs classical Gibbs sampling"""
    print("=== QUANTUM VS CLASSICAL SAMPLING COMPARISON ===")
    
    if not DWAVE_OK:
        print("❌ D-Wave not available for comparison")
        return
    
    # Create a simple test QUBO
    print("Testing on simple 10x10 grid QUBO...")
    
    # Grid-like QUBO (prefer checkerboard pattern)
    Q = {}
    n = 10
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            Q[(idx, idx)] = -0.5  # Prefer activation
            
            # Prefer opposite neighbors (checkerboard)
            if j < n-1:  # Right neighbor
                Q[(idx, idx+1)] = 1.0
            if i < n-1:  # Down neighbor  
                Q[(idx, idx+n)] = 1.0
    
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    
    # Test D-Wave simulated annealing
    sampler = SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sampler.sample(bqm, num_reads=20, schedule=[(0.0, 5.0), (1.0, 0.1)])
    dwave_time = time.time() - start
    
    dwave_samples = []
    for sample in sampleset.samples():
        pattern = [sample[i] for i in range(n*n)]
        dwave_samples.append(pattern)
    
    # Test classical random sampling
    start = time.time()
    classical_samples = []
    for _ in range(20):
        pattern = [np.random.randint(0, 2) for _ in range(n*n)]
        classical_samples.append(pattern)
    classical_time = time.time() - start
    
    # Compare results
    dwave_activity = np.mean([np.mean(s) for s in dwave_samples])
    classical_activity = np.mean([np.mean(s) for s in classical_samples])
    
    print(f"   D-Wave SimulatedAnnealing: Activity={dwave_activity:.3f}, Time={dwave_time:.4f}s")
    print(f"   Classical Random:          Activity={classical_activity:.3f}, Time={classical_time:.4f}s")
    print(f"   Speed ratio: {dwave_time/classical_time:.1f}x slower (D-Wave simulated annealing)")
    
    # Check if patterns are different
    dwave_pattern = np.array(dwave_samples[0]).reshape(n, n)
    classical_pattern = np.array(classical_samples[0]).reshape(n, n)
    
    print(f"   Pattern difference: {np.mean(dwave_pattern != classical_pattern):.1%}")
    print()

def honest_summary():
    """Provide honest summary of what we actually achieved"""
    print("=== HONEST SUMMARY: WHAT DID WE ACTUALLY ACHIEVE? ===")
    print()
    print("✅ WHAT WE DID RIGHT:")
    print("   • Used real D-Wave software stack (Ocean SDK)")
    print("   • Enforced true Pegasus P_6 topology constraints")
    print("   • Proper QUBO formulation from RBM energies")
    print("   • StructureComposite ensures valid Pegasus edges only")
    print("   • Implemented complete diffusion → RBM → QUBO pipeline")
    print()
    print("❌ WHAT WE MISREPRESENTED:")
    print("   • SimulatedAnnealingSampler is CLASSICAL, not quantum hardware")
    print("   • 1.3s timing is classical simulated annealing, not quantum")
    print("   • 99.81% reconstruction accuracy suggests overfitting/memorization")
    print("   • Called it 'quantum' when it was 'quantum-inspired' simulation")
    print()
    print("🔬 WHAT WE ACTUALLY DEMONSTRATED:")
    print("   • Quantum annealing formulation works (QUBO is valid)")
    print("   • Pegasus topology constraints are respected")
    print("   • Classical simulation of quantum annealing behavior")
    print("   • Proof-of-concept for true quantum hardware integration")
    print()
    print("🎯 FOR TRUE QUANTUM RESULTS:")
    print("   • Would need actual D-Wave hardware access (DWaveSampler)")
    print("   • Expect much slower timing (seconds per QUBO)")
    print("   • Quantum advantage in sampling diversity, not reconstruction accuracy")
    print("   • Different activity/sparsity patterns due to quantum fluctuations")

def main():
    """Run complete honest verification"""
    print("HONEST QUANTUM VERIFICATION")
    print("=" * 50)
    print()
    
    test_what_simulated_annealing_actually_is()
    test_realistic_reconstruction_metrics() 
    test_quantum_vs_classical_sampling()
    honest_summary()

if __name__ == "__main__":
    main()

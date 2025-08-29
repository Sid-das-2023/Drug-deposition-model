# Physics-Informed Neural Network (PINN) for Airway Drug Deposition

## Overview

This repository implements a **Physics-Informed Neural Network (PINN)** to simulate drug particle deposition in human airways. The model combines deep learning with fundamental physics equations to solve complex biomedical problems in respiratory medicine and pharmaceutical drug delivery.

## Table of Contents

1. [Problem Description](#problem-description)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Particle Physics Theories](#particle-physics-theories)
4. [Neural Network Architecture](#neural-network-architecture)
5. [Training Methodology](#training-methodology)
6. [Geometry and Domain](#geometry-and-domain)
7. [Physical Parameters](#physical-parameters)
8. [Visualization](#visualization)
9. [Applications](#applications)
10. [Installation and Usage](#installation-and-usage)

## Problem Description

### Application Domain
This PINN model addresses critical challenges in:
- **Inhaler Design**: Optimizing drug delivery devices for respiratory diseases
- **Pharmaceutical Development**: Understanding particle deposition patterns for different drug formulations
- **Respiratory Disease Research**: Modeling airflow and particle transport in diseased airways
- **Personalized Medicine**: Patient-specific airway modeling for targeted therapy

### Physical System
The model simulates:
- **Airflow dynamics** in simplified 2D airway geometries
- **Drug particle transport** including convection, diffusion, and deposition
- **Multi-physics coupling** between fluid flow and particle dynamics

## Mathematical Foundation

### Governing Equations

#### 1. Navier-Stokes Equations (Fluid Flow)

**Continuity Equation (Mass Conservation):**
```
∇ · u = ∂u_x/∂x + ∂u_y/∂y = 0
```

**Momentum Equations:**
```
ρ(u · ∇)u + ∇p = μ∇²u
```

Expanded form:
- **X-momentum:** `ρ(u_x ∂u_x/∂x + u_y ∂u_x/∂y) + ∂p/∂x = μ(∂²u_x/∂x² + ∂²u_x/∂y²)`
- **Y-momentum:** `ρ(u_x ∂u_y/∂x + u_y ∂u_y/∂y) + ∂p/∂y = μ(∂²u_y/∂x² + ∂²u_y/∂y²)`

#### 2. Convection-Diffusion Equation (Particle Transport)

For particle concentration C:
```
∂C/∂t + u · ∇C = D_p ∇²C
```

Where:
- `∂C/∂t + u_x ∂C/∂x + u_y ∂C/∂y = D_p(∂²C/∂x² + ∂²C/∂y²)`

## Particle Physics Theories

### 1. Cunningham Slip Correction Factor
```python
Cc = 1 + (2λ/d_p)[A1 + A2·exp(-A3·d_p/(2λ))]
```
- **Purpose**: Corrects Stokes drag for small particles where continuum assumption breaks down
- **Constants**: A1=1.257, A2=0.4, A3=1.1 (empirical values)
- **λ**: Mean free path of air molecules (~65 nm)

### 2. Brownian Diffusivity (Einstein-Stokes Equation)
```python
D_p = (k_B·T·Cc)/(3π·μ·d_p)
```
- **Theory**: Random molecular motion causes particle diffusion
- **k_B**: Boltzmann constant (1.380649×10⁻²³ J/K)
- **T**: Temperature (310K, body temperature)

### 3. Particle Relaxation Time (Stokes Law)
```python
τ_p = (ρ_p·d_p²·Cc)/(18μ)
```
- **Theory**: Time for particle to adjust to flow changes
- **Critical for**: Impaction mechanisms in respiratory deposition

### 4. Settling Velocity (Gravitational)
```python
v_g = ((ρ_p - ρ)·g·d_p²·Cc)/(18μ)
```
- **Theory**: Terminal velocity under gravity
- **Buoyancy correction**: (ρ_p - ρ) accounts for fluid density difference

### 5. Deposition Velocity (Composite Model)
```python
v_dep = k_diff + k_sed + k_imp
```
Combines three deposition mechanisms:
1. **Diffusional**: `k_diff = Sh·D_p/D_h` (Brownian motion)
2. **Sedimentation**: `k_sed = v_g` (gravitational settling)
3. **Impaction**: `k_imp = χ·U·Stk/D_h` (inertial impaction)

## Neural Network Architecture

### Network Structure

#### 1. Fourier Feature Encoding
```python
def fourier_features(x, n_freq=6):
    freqs = 2^k * π, k = 0,1,2,...,n_freq-1
    return [sin(freq·x), cos(freq·x)] for all frequencies
```
- **Purpose**: Enables neural networks to learn high-frequency functions
- **Theory**: Based on harmonic analysis for better approximation of oscillatory solutions

#### 2. Multi-Layer Perceptron (MLP)
- **Architecture**: Deep feedforward network with Tanh activation
- **Initialization**: Xavier uniform (optimal for Tanh activation)
- **Dimensions**: 256 neurons width, 8 layers depth

#### 3. PINN Architecture
```
Input Features: [x, y, t, conditions, fourier_features]
         ↓
Main Model (MLP) → [u_x, u_y, p, C]
         ↓
Separate Head (MLP) → [m_s] (wall deposition rate)
```

### Input and Output Variables

#### Input Features
- **Spatial coordinates**: (x, y) - position in airway
- **Time**: t - temporal evolution
- **Conditions**: [Q, U_in, d_p, ΔP] - breathing parameters
- **Fourier features**: Encoded (x,y,t) for enhanced learning

#### Output Variables
- **u = [u_x, u_y]**: Velocity field components (m/s)
- **p**: Pressure field (Pa)
- **C**: Particle concentration (dimensionless)
- **m_s**: Wall deposition rate (kg/m²·s)

## Training Methodology

### Loss Function Components

#### 1. Physics-Informed Loss
```python
L_physics = L_continuity + L_momentum_x + L_momentum_y + L_convection_diffusion
```
Each term enforces the respective PDE to be satisfied at collocation points.

#### 2. Boundary Condition Loss

**Inlet Boundary Conditions:**
```python
u_x = U_in * (1 - (y/R)²)  # Parabolic velocity profile (Poiseuille flow)
u_y = 0                     # No cross-flow
C = 1                       # Unit concentration
```

**Wall Boundary Conditions:**
```python
u_x = u_y = 0  # No-slip condition (zero velocity at walls)
```

#### 3. Total Loss Function
```python
L_total = L_physics + 10.0 * (L_inlet + L_wall)
```
The weighting factor (10.0) ensures strong enforcement of boundary conditions.

### Optimization Strategy
- **Optimizer**: Adam with learning rate 1×10⁻³
- **Scheduler**: Exponential decay (γ=0.995)
- **Training iterations**: 1000 (adjustable)

## Geometry and Domain

### Simplified Airway Model
- **Length**: L = 6 cm (representative of tracheal scale)
- **Half-height**: R = 1 cm (circular cross-section radius)
- **Shape**: 2D rectangular channel approximation
- **Coordinate system**: x ∈ [0, L], y ∈ [-R, R]

### Sampling Strategy
- **Domain points**: Random sampling inside the channel (N=1000)
- **Boundary points**: Structured sampling at inlet, outlet, and walls (N=200)
- **Time window**: t ∈ [0, 0.1] seconds (respiratory cycle fraction)

## Physical Parameters

### Fluid Properties (Air at Body Temperature)
```python
ρ = 1.2 kg/m³          # Air density
μ = 1.8×10⁻⁵ Pa·s      # Dynamic viscosity
T = 310 K              # Body temperature (37°C)
```

### Particle Properties
```python
ρ_p = 1200 kg/m³       # Drug particle density
d_p = 3 μm             # Particle diameter
λ = 65 nm              # Air mean free path
```

### Flow Conditions
```python
Q = 30 L/min           # Breathing rate (5×10⁻⁴ m³/s)
U_in = 2 m/s           # Mean inlet velocity
ΔP = 100 Pa            # Pressure drop across airway
```

### Physical Constants
```python
k_B = 1.380649×10⁻²³ J/K  # Boltzmann constant
g = 9.81 m/s²             # Gravitational acceleration
```

## Visualization

The model generates four key visualization plots:

1. **Velocity Magnitude**: Flow speed distribution throughout the airway
2. **Pressure Field**: Pressure gradients driving the flow
3. **Particle Concentration**: Spatial distribution of drug particles
4. **Velocity Vectors**: Flow direction and circulation patterns

### Visualization Features
- **Grid resolution**: 50×30 points for detailed visualization
- **Color maps**: Viridis, Plasma, and Cool for different fields
- **Vector field**: Subsampled quiver plot for flow visualization

## Applications

### Medical and Pharmaceutical Applications
- **Drug Design**: Optimizing particle size for targeted lung deposition
- **Inhaler Development**: Improving delivery device efficiency and design
- **Disease Modeling**: Understanding pathological flow patterns in respiratory diseases
- **Personalized Medicine**: Patient-specific airway modeling for individualized therapy

### Research Applications
- **Computational Fluid Dynamics**: Validating traditional CFD approaches
- **Machine Learning in Physics**: Demonstrating physics-informed neural networks
- **Biomedical Engineering**: Advancing respiratory system modeling techniques

## Installation and Usage

### Prerequisites
```bash
pip install torch torchvision matplotlib numpy
```

### Running the Model

1. **Execute the Jupyter notebook**: `pinndeposition.ipynb`
2. **Install dependencies**: Run the first cell to install required packages
3. **Train the model**: Execute the main training cell
4. **Visualize results**: Run the visualization cells

### Key Files
- `pinndeposition.ipynb`: Complete PINN implementation with training and visualization
- `README.md`: This comprehensive documentation

### Model Execution Flow
1. **Setup**: Initialize device, constants, and neural network
2. **Sampling**: Generate collocation and boundary points
3. **Training**: Optimize physics-informed loss function
4. **Validation**: Visualize predicted flow and concentration fields
5. **Analysis**: Interpret results for drug deposition patterns

## Advanced Features

### Technical Innovations
- **Multi-physics coupling**: Simultaneous fluid flow and particle transport
- **Automatic differentiation**: Exact gradient computation for PDE enforcement
- **Condition-dependent modeling**: Adapts to different breathing patterns and particle properties
- **Wall deposition tracking**: Monitors drug accumulation at airway surfaces

### Computational Advantages
- **Mesh-free approach**: No grid generation required
- **Continuous solutions**: Smooth predictions at any spatial-temporal location
- **Physics enforcement**: Hard constraints ensure physical consistency
- **Efficient inference**: Fast prediction once trained

## Future Enhancements

### Potential Improvements
1. **3D geometry**: Extension to realistic airway geometries
2. **Multiple particle sizes**: Polydisperse aerosol modeling
3. **Turbulent flows**: Reynolds-averaged Navier-Stokes equations
4. **Mucus layer effects**: Including realistic airway surface conditions
5. **Breathing dynamics**: Time-varying boundary conditions

### Research Directions
- **Inverse problems**: Parameter estimation from experimental data
- **Uncertainty quantification**: Bayesian neural networks for uncertainty
- **Multi-scale modeling**: Coupling with alveolar deposition models
- **Clinical validation**: Comparison with in-vivo deposition measurements

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pinn_airway_deposition_2025,
  title={Physics-Informed Neural Network for Airway Drug Deposition},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/Drug-deposition-model}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaborations, please contact:
- **Email**: [your-email@domain.com]
- **GitHub**: [your-github-username]

---

*This README provides a comprehensive overview of the PINN implementation for airway drug deposition modeling. The code represents an intersection of machine learning, computational fluid dynamics, and biomedical engineering for advancing respiratory medicine research.*

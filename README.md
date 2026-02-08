# Gradient-Based Optimization Algorithms 🎯
This project implements and visualizes six different gradient-based optimization algorithms:

- **Gradient Descent (GD)**: The foundational algorithm that uses the full dataset gradient for each update. Guaranteed convergence for convex functions with proper learning rate, but computationally expensive.

- **Stochastic Gradient Descent (SGD)**: Uses single random samples for updates. Much faster but noisier convergence. Can escape local minima due to inherent stochasticity.

- **Mini-Batch Gradient Descent**: The practical middle ground - uses small batches of data. Balances computational efficiency with gradient estimate quality. Most widely used in practice.

- **Adam (Adaptive Moment Estimation)**: Maintains adaptive learning rates for each parameter using first and second moment estimates. Currently the most popular optimizer in deep learning.

- **Nesterov Accelerated Gradient (NAG)**: Improved momentum method that "looks ahead" by computing gradients at anticipated future positions. Better convergence properties than standard momentum.

- **Adan (Adaptive Nesterov Momentum)**: Recent optimizer combining adaptive learning rates with Nesterov momentum and gradient differences. Shows faster convergence than Adam in many scenarios.

## What will you find in the code?

- Six different optimization algorithms implemented from scratch
- Three classic test functions (Rosenbrock, Beale, Sphere) but you can also test it with your own function
- Parameter selection
- Visualization of optimization paths
- Convergence analysis and comparison
- Gradient magnitude tracking
- Performance metrics and statistics
- Mathematical theory documentation

## Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
```


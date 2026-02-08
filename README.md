# Gradient-Based Optimization Algorithms 🎯
This project implements and visualizes six different gradient-based optimization algorithms:

- **Gradient Descent (GD)**: The foundational algorithm that uses the full dataset gradient for each update. Guaranteed convergence for convex functions with proper learning rate, but computationally expensive.

- **Stochastic Gradient Descent (SGD)**: Uses single random samples for updates. Much faster but noisier convergence. Can escape local minima due to inherent stochasticity.

- **Mini-Batch Gradient Descent**: The practical middle ground - uses small batches of data. Balances computational efficiency with gradient estimate quality. Most widely used in practice.

- **Adam (Adaptive Moment Estimation)**: Maintains adaptive learning rates for each parameter using first and second moment estimates. Currently the most popular optimizer in deep learning.

- **Nesterov Accelerated Gradient (NAG)**: Improved momentum method that "looks ahead" by computing gradients at anticipated future positions. Better convergence properties than standard momentum.

- **Adan (Adaptive Nesterov Momentum)**: Recent optimizer combining adaptive learning rates with Nesterov momentum and gradient differences. Shows faster convergence than Adam in many scenarios.

## What will you find in the code?

- Implementation of 6 optimization algorithms with full mathematical foundations
- Interactive function selection (predefined or custom)
- Automatic gradient computation for custom functions
- Comprehensive visualization suite:
  - Optimization paths on contour plots
  - Individual optimizer trajectories
  - Convergence curves (log scale)
  - Gradient magnitude evolution
  - Side-by-side comparisons
- Quantitative performance metrics for each optimizer
- Divergence detection and handling
- Gradient clipping for numerical stability

This project was developed as an educational tool to:
- Understand how different optimization algorithms behave
- Compare convergence speed and stability
- Visualize optimization trajectories on complex landscapes
- Study the impact of hyperparameters
- Demonstrate practical considerations in optimizer selection
  
## Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
```

### Required Inputs

The script will prompt you to enter:

1. **Choose a function to optimize**

2. **Starting point**: Two numeric values for initial position `(x, y)`

3. **Learning rate (η)**: Step size for gradient descent (e.g., 0.01, 0.001)

4. **Number of iterations**: Maximum optimization steps (e.g., 1000, 5000)

5. **Batch size** (for Mini-Batch GD): Number of samples per batch (e.g., 32)

### Example Test Functions

The project includes three classic optimization test functions:

#### Rosenbrock Function 
*(Non-convex with narrow valley, global minimum at (1, 1))*

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

- Global minimum: $f(1, 1) = 0$
- Challenging due to narrow, curved valley
- Good starting point: `(-1.5, 2.5)`
- Recommended learning rate: `0.001`

#### Beale Function 
*(Multiple local minima, tests exploration capability)*

$$f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$$

- Global minimum: $f(3, 0.5) = 0$
- Multiple local minima
- Good starting point: `(1, 1)`
- Recommended learning rate: `0.001`

#### Sphere Function 
*(Simple convex quadratic, easy to optimize)*

$$f(x, y) = x^2 + y^2$$

- Global minimum: $f(0, 0) = 0$
- Perfectly convex
- Good starting point: `(5, 5)`
- Recommended learning rate: `0.1`

### Custom Functions

For custom functions, use NumPy syntax. 

##  Output

The program generates:

### 1. Visual Plots

**Optimization Paths on Contour Plots**:
- All optimizers overlaid on the objective function
- Individual trajectory plots for each optimizer
- Starting points (black star ★) and ending points (colored circles)
- Function contours showing the optimization landscape

**Convergence Analysis**:
- Log-scale function value vs iteration
- Gradient magnitude evolution
- Side-by-side comparisons of all optimizers

### 2. Numerical Metrics

For each optimizer, the script reports:
- **Final position**: Coordinates `(x, y)` after optimization
- **Final function value**: $f(x, y)$ at convergence
- **Minimum value reached**: Best value encountered during optimization
- **Final gradient norm**: $\|\nabla f\|$ at final position
- **Iterations to threshold**: Steps needed to reach $f(x) < 1.0$
- **Total iterations**: Actual steps completed (may be less if diverged)
- **Divergence warnings**: Alerts if optimizer became unstable

### 3. Comparative Summary

- Convergence speed comparison
- Stability analysis (variance in trajectory)
- Final accuracy achieved by each method
- Practical recommendations based on results

## Theory

### General Optimization Problem

We aim to minimize a function $f(\theta)$ where $\theta = (x, y)$ represents the parameters:

$$\min_{\theta} f(\theta)$$

---

### Gradient Descent (GD)

The basic update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$$

where:
- $\theta_t$ = parameter vector at iteration $t$
- $\eta$ = learning rate (step size)
- $\nabla f(\theta_t)$ = gradient of $f$ at $\theta_t$

**Characteristics**:
- Uses entire dataset (or full gradient)
- Guaranteed convergence for convex functions with proper $\eta$
- Can be slow for large datasets

---

### Stochastic Gradient Descent (SGD)

Uses a single randomly selected sample:

$$\theta_{t+1} = \theta_t - \eta \nabla f_i(\theta_t)$$

where $f_i$ is the loss for the $i$-th sample.

**Characteristics**:
- Much faster updates than GD
- High variance leads to noisy convergence
- Can escape local minima due to noise
- Requires careful learning rate tuning

---

### Mini-Batch Gradient Descent

Uses a subset (mini-batch) $B$ of the data:

$$\theta_{t+1} = \theta_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla f_i(\theta_t)$$

where $|B|$ is the batch size.

**Characteristics**:
- Balances GD and SGD
- Reduces variance compared to SGD
- Leverages vectorization for computational efficiency
- Most common in practice

---

### Nesterov Accelerated Gradient (NAG)

Looks ahead before computing gradient:

$$v_{t+1} = \beta v_t + \nabla f(\theta_t - \eta \beta v_t)$$

$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

where $\beta$ is the momentum coefficient (typically $\beta = 0.9$).

**Characteristics**:
- Better convergence than standard momentum
- "Looks ahead" to anticipated position
- Particularly effective for convex functions
- Reduces oscillations in narrow valleys

---

### Adam (Adaptive Moment Estimation)

Maintains running averages of gradient and squared gradient:

**First moment (momentum)**:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla f(\theta_t)$$

**Second moment (adaptive learning rate)**:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)[\nabla f(\theta_t)]^2$$

**Bias correction**:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update rule**:
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Characteristics**:
- Adaptive per-parameter learning rates
- Combines benefits of momentum and RMSprop
- Works well with sparse gradients
- Requires minimal hyperparameter tuning
- Most popular choice in deep learning

---

### Adan (Adaptive Nesterov Momentum)

Combines adaptive learning with Nesterov momentum using gradient differences:

**First moment**:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla f(\theta_t)$$

**Gradient difference (Nesterov component)**:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)[\nabla f(\theta_t) - \nabla f(\theta_{t-1})]$$

**Second moment**:
$$n_t = \beta_3 n_{t-1} + (1 - \beta_3)[\nabla f(\theta_t) + \beta_2(\nabla f(\theta_t) - \nabla f(\theta_{t-1}))]^2$$

**Update rule**:
$$\theta_{t+1} = \theta_t - \eta \frac{m_t + \beta_2 v_t}{\sqrt{n_t} + \epsilon}$$

Typical values: $\beta_1 = 0.98$, $\beta_2 = 0.92$, $\beta_3 = 0.99$, $\epsilon = 10^{-8}$

**Characteristics**:
- Often faster convergence than Adam
- More stable training dynamics
- Effective for both convex and non-convex problems
- Newer algorithm (2023) showing promising results

---

## Key Observations

Based on typical results across different test functions:

### 1. Convergence Speed
- **Adam** and **Adan** typically converge fastest due to adaptive learning rates
- **Nesterov** shows improved convergence over basic momentum
- **SGD** is noisy but can explore the space effectively
- **GD** and **Mini-Batch GD** are more conservative but reliable

### 2. Stability
- **GD** and **Mini-Batch GD**: Smooth, stable convergence paths
- **SGD**: High variance, oscillatory behavior, but good exploration
- **Adam/Adan**: Balance between speed and stability
- **Nesterov**: Reduced oscillations compared to standard momentum

### 3. Final Accuracy
- All methods can reach similar final values with proper hyperparameter tuning
- Adaptive methods (Adam, Adan) are more robust to learning rate selection
- Non-adaptive methods require more careful tuning but can be more precise

### 4. Practical Considerations
- **Mini-Batch GD**: Most common in practice (good speed/stability trade-off)
- **Adam**: Default choice for most deep learning applications
- **Adan**: Promising for research and when faster convergence is critical
- **SGD with momentum**: Still competitive with proper tuning
- **Nesterov**: Excellent for convex optimization problems

### 5. Function-Specific Behavior
- **Rosenbrock**: Adaptive methods handle the narrow valley better
- **Beale**: All methods struggle with multiple local minima; SGD's noise can help
- **Sphere**: All methods perform well; differences in convergence speed are most visible

## Implementation Details

### Gradient Clipping
All optimizers implement gradient clipping to prevent divergence:
- Maximum gradient norm: 10.0
- Applied before parameter updates
- Prevents numerical instability in steep regions

### Numerical Gradient Computation
For custom functions, gradients are computed using central differences:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}$$

with $h = 10^{-5}$

### Divergence Detection
The script monitors for:
- NaN (Not a Number) values
- Infinite values
- Stops optimization early if detected
- Reports partial results

## Limitations and Notes

### Numerical Considerations
- High learning rates can cause divergence (especially for GD, SGD, Nesterov)
- Very low learning rates lead to slow convergence
- Gradient clipping helps but may slow convergence near boundaries
- Floating-point precision limits accuracy near minima

### Algorithm-Specific
- **SGD**: Single-sample simulation (not truly stochastic without data batches)
- **Mini-Batch GD**: Batch size affects memory and convergence
- **Adam/Adan**: More hyperparameters to tune (though defaults often work well)
- **Nesterov**: Momentum parameter $\beta$ significantly impacts behavior

### Function-Specific
- Algorithms perform differently on convex vs non-convex functions
- Ill-conditioned problems (extreme curvature differences) challenge all methods
- Multi-modal functions may trap optimizers in local minima
- Some functions may require function-specific tuning

### General
- 2D visualization only (algorithms work in any dimension)
- No line search or adaptive learning rate schedules
- No regularization or constraints
- Results depend heavily on hyperparameter choices

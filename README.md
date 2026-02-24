What the script does

Presents a runtime menu of predefined problems — user selects which function to solve
Uses SymPy to automatically derive the Laplacian of any chosen expression — no manual calculus required
Converts the symbolic result through a SymPy → NumPy → PyTorch pipeline into training-ready functions
Bundles all problem-specific information (f_source, g_boundary, u_true) into a clean ProblemConfig object
Trains a 5-layer fully connected network with tanh activations for 5000 epochs
Each epoch samples 1000 random interior points and 1000 random boundary points

Enforces physics via two loss terms:
1. Interior loss — network's Laplacian must match f_source at interior points
2. Boundary loss — network's predictions must match g_boundary on edges (weighted 10×)

After training, evaluates on a held-out 100×100 grid of 10,000 unseen points
Reports validation metrics:
1. Mean Absolute Error (MAE)
2. Maximum Absolute Error
3. L2 Relative Error

Produces 6-panel visualizations: 3D surface plots and 2D heatmaps of the true solution, PINN prediction, and absolute error

Predefined Functions Include: 
Trig: sin(πx)sin(πy)
Linear: xy
Quadratic: x² + y²
Polynomial: x(1-x)y(1-y)
Sqrt: √(1 + x² + y²)
Exponential: eˣ cos(πy)
Cubic: x³ + xy²
Log: log(0.001 + x² + y²)


Why this matters: 
PINNs learn purely from physics constraints — the network never sees the true solution during training. It only checks if its output satisfies ∇²u = f and matches boundary conditions. The mathematics guarantees these two conditions uniquely determine the correct solution.
The SymPy auto-differentiation pipeline means this framework generalizes to any smooth function without any manual calculus — a significant upgrade over hardcoded single-function implementations.
Random resampling of collocation points each epoch prevents overfitting and forces the network to generalize across the entire domain.
The ProblemConfig abstraction keeps the training loop identical across all problems — swap the problem, not the code.
Validation on a held-out uniform grid provides a genuine measure of generalization, not just training performance.


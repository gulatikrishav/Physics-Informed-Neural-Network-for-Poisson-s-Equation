# Extends the original single-problem PINN to support dynamic problem selection.
# The key upgrade: instead of hardcoding f_source and u_true for sin(πx)sin(πy),
# we use SymPy to symbolically derive the Laplacian from ANY user-supplied expression.
# This means adding a new problem requires only one line, not manual calculus.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# ==============================================================================
# PART 0: DEVICE
# ==============================================================================

# Checks if a GPU is available at runtime — if yes use it (much faster),
# if not fall back to CPU. The if/else is necessary because the code doesn't
# know in advance what machine it'll run on. Without this check, forcing GPU
# on a machine without one would just crash.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==============================================================================
# PART 1: NETWORK DEFINITION
# ==============================================================================
# THIS IS JUST THE BLUEPRINT — not where learning happens.
# Think of this like drawing the structure of the network before anything gets built.
# The actual learning happens later in the training loop (Part 6).
#
# Architecture: 2 → 50 → 50 → 50 → 50 → 1
# - 1 input layer (takes x,y coordinates)
# - 3 hidden layers (learn increasingly complex patterns)
# - 1 output layer (outputs a single predicted value u(x,y))

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Expand: lift 2D coordinates into 50D feature space
        self.fc1 = nn.Linear(2, 50)
        # Transform: refine features through multiple layers
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        # Collapse: reduce to single solution value
        self.fc5 = nn.Linear(50, 1)

    def forward(self, x, y):
        # Combine x and y into one [N, 2] matrix — each row is one (x,y) point
        inputs = torch.cat([x, y], dim=1)

        # Data flows through each layer using matrix multiplication.
        # pointless because the math would collapse into one linear operation.
        # tanh introduces the non-linearity needed to approximate complex surfaces.
        # Why tanh specifically: outputs between -1 and 1 (prevents exploding values),
        # smooth and differentiable everywhere (needed for gradient computation).
        out = torch.tanh(self.fc1(inputs))  # Basic spatial patterns
        out = torch.tanh(self.fc2(out))     # Combine into complex features
        out = torch.tanh(self.fc3(out))     # Further refinement
        out = torch.tanh(self.fc4(out))     # Final feature processing

        # No activation on final layer — we want the raw predicted scalar value u(x,y)
        # This single number is the network's prediction of the PDE solution at that point
        out = self.fc5(out)

        return out


# ==============================================================================
# PART 2: SAMPLING FUNCTIONS
# ==============================================================================
# We can't train on every possible point (there are infinitely many).
# Instead we randomly sample a subset each epoch. If the network satisfies
# the PDE at enough random points, it generalizes to the whole domain.

def sample_interior(n_points, device=device):
    # Generate 1000 random (x,y) points inside the unit square [0,1]×[0,1]
    # requires_grad=True is critical — it tells PyTorch to track these values
    # so we can later differentiate the network's output w.r.t. x and y
    # (needed to compute the Laplacian in the loss function)
    x = torch.rand(n_points, 1, device=device).requires_grad_(True)
    y = torch.rand(n_points, 1, device=device).requires_grad_(True)
    return x, y


def sample_boundary(n_points_per_edge, device=device):
    # Boundary points can't be fully random — they must sit exactly on the edges.
    # So for each edge, one coordinate is FIXED (pinned to the edge) and
    # one is RANDOM (varies along that edge).
    #
    # Important distinction: the LOCATIONS are fixed to the edges (geometry never changes),
    # but the VALUES the network must predict at those locations change based on
    # which function was chosen (handled by g_boundary in ProblemConfig).
    # e.g. at point (0.5, 0) on the bottom edge:
    #   sin(πx)sin(πy) → value is 0
    #   x**2 + y**2   → value is 0.25

    x_bottom = torch.rand(n_points_per_edge, 1, device=device)
    y_bottom = torch.zeros(n_points_per_edge, 1, device=device)  # y fixed at 0

    x_top = torch.rand(n_points_per_edge, 1, device=device)
    y_top = torch.ones(n_points_per_edge, 1, device=device)      # y fixed at 1

    x_left = torch.zeros(n_points_per_edge, 1, device=device)    # x fixed at 0
    y_left = torch.rand(n_points_per_edge, 1, device=device)

    x_right = torch.ones(n_points_per_edge, 1, device=device)    # x fixed at 1
    y_right = torch.rand(n_points_per_edge, 1, device=device)

    # Stack all four edges into one combined list of boundary points
    # Total = 4 × 250 = 1000 boundary points per epoch
    x_boundary = torch.cat([x_bottom, x_top, x_left, x_right], dim=0)
    y_boundary = torch.cat([y_bottom, y_top, y_left, y_right], dim=0)
    return x_boundary, y_boundary


# ==============================================================================
# PART 3: DYNAMIC PROBLEM DEFINITION (SYMPY -> TORCH)
# ==============================================================================

# ProblemConfig is just a container — no math happens here.
# It bundles the three things any PDE problem needs into one neat package:
#   1. u_true  — what is the true solution? (used after training to check accuracy)
#   2. f_source — what is the right hand side of ∇²u = f? (used in interior loss)
#   3. g_boundary — what are the correct boundary values? (used in boundary loss)
#
# By bundling these together, the training loop stays identical regardless of
# which problem was chosen — it just receives this package and uses it.
class ProblemConfig:
    def __init__(self, name, u_true, f_source, g_boundary):
        self.name = name
        self.u_true = u_true          # Analytical solution (evaluation only)
        self.f_source = f_source      # RHS of ∇²u = f (used in interior loss)
        self.g_boundary = g_boundary  # Boundary values (used in boundary loss)


def create_problem_from_expression(expression_str, name="Custom"):
    # This is the heart of the refactor — SymPy does the calculus automatically.
    # In the original code, adding a new u(x,y) required manually computing
    # the Laplacian by hand. Here, SymPy does it for you.
    #
    # Full pipeline: String → SymPy (calculus) → NumPy (numbers) → PyTorch (training)
    # Each library hands off to the next because each does something the others can't:
    #   SymPy  — does symbolic calculus (can't work with data arrays)
    #   NumPy  — evaluates math numerically (can't do deep learning)
    #   PyTorch — runs the network and computes gradients (can't do symbolic math)

    # Create symbolic variables — not numbers, actual algebraic symbols like on paper
    x_sym, y_sym = sp.symbols("x y")

    # eval() turns the string "x**2 + y**2" into a real SymPy expression.
    # safe_globals restricts what's accessible — a security measure so only
    # valid math expressions are allowed, not arbitrary Python code.
    safe_globals = {"sp": sp, "__builtins__": {}}
    safe_locals = {"x": x_sym, "y": y_sym}
    u_symbolic = sp.simplify(eval(expression_str, safe_globals, safe_locals))

    # This single line replaces doing calculus by hand.
    # sp.diff(u, x, 2) computes ∂²u/∂x² symbolically — exact, not approximate.
    # Adding the y version gives the full Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y²
    laplacian_symbolic = sp.simplify(
        sp.diff(u_symbolic, x_sym, 2) + sp.diff(u_symbolic, y_sym, 2)
    )

    # lambdify converts SymPy expressions into callable NumPy functions.
    # Before this, u_symbolic is just algebra. After this, u_func_np(0.5, 0.3)
    # gives you an actual number.
    u_func_np = sp.lambdify((x_sym, y_sym), u_symbolic, "numpy")
    lap_func_np = sp.lambdify((x_sym, y_sym), laplacian_symbolic, "numpy")

    def _to_torch_same_shape(result_np, ref_tensor):
        # Edge case: when the Laplacian is a constant (e.g. for x**2 + y**2,
        # ∇²u = 4), lambdify returns a single scalar instead of an array.
        # PyTorch needs a full array matching the batch of points.
        # This function detects that and broadcasts the scalar to the right shape.
        arr = np.array(result_np, dtype=float)
        if arr.shape == ():
            arr = np.full(ref_tensor.shape, float(arr), dtype=float)
        else:
            arr = np.broadcast_to(arr, ref_tensor.shape)
        return torch.from_numpy(arr.copy()).float().to(ref_tensor.device)

    def u_true_torch(x, y):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        return _to_torch_same_shape(u_func_np(x_np, y_np), x)

    def f_source_torch(x, y):
        # The RHS of ∇²u = f, auto-derived by SymPy.
        # This is what the network's Laplacian gets compared against during training.
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        return _to_torch_same_shape(lap_func_np(x_np, y_np), x)

    def g_boundary_torch(x, y):
        # The correct boundary values for whichever function was chosen.
        # The original code always used g=0 because sin(πx)sin(πy) happens to
        # vanish on all edges. For general functions this isn't true, so we
        # use the true solution's actual boundary values instead.
        return u_true_torch(x, y)

    full_name = f"{name}: u(x,y) = {u_symbolic}"
    print(f"\nCreated problem '{name}'")
    print(f"  u(x,y)      = {u_symbolic}")
    print(f"  ∇²u (auto)  = {laplacian_symbolic}")

    return ProblemConfig(full_name, u_true_torch, f_source_torch, g_boundary_torch)


# ==============================================================================
# PART 3.5: PROBLEM MENU (USER CHOICE)
# ==============================================================================
# To add a new function (e.g. Gaussian), just add one line here:
# "7": ("gaussian", "sp.exp(-(x**2 + y**2))", "Gaussian")
# SymPy handles the Laplacian automatically — no calculus by hand required.

PREDEFINED = {
    "1": ("trig",      "sp.sin(sp.pi*x)*sp.sin(sp.pi*y)", "Trigonometric"),
    "2": ("linear",    "x*y",                               "Linear"),
    "3": ("quadratic", "x**2 + y**2",                       "Quadratic"),
    "4": ("poly",      "x*(1-x)*y*(1-y)",                   "Polynomial"),
    "5": ("sqrt",      "sp.sqrt(1 + x**2 + y**2)",          "SquareRoot (radical)"),
    "6": ("exp",       "sp.exp(x) * sp.cos(sp.pi*y)",       "ExpTrig"),
    "7": ("cubic",     "x**3 + x*y**2",                     "Cubic"),
    "8": ("log",       "sp.log (0.001 + x**2 + y**2)",           "Logarithmic"),
}

def choose_problem():
    # Presents the menu, takes the user's input, and passes the chosen
    # expression string to create_problem_from_expression.
    # Returns a fully built ProblemConfig ready for training.
    # The training loop never changes — only the ProblemConfig does.
    print("\n" + "="*70)
    print("SELECT A PROBLEM TO SOLVE (enter a number):")
    print("="*70)
    for k in sorted(PREDEFINED.keys(), key=int):
        keyname, expr, title = PREDEFINED[k]
        print(f"{k}. {title:<24}  (key: '{keyname}')   u = {expr}")
    print("="*70)

    choice = input("Your choice: ").strip()

    if choice in PREDEFINED:
        keyname, expr, title = PREDEFINED[choice]
    else:
        matches = [v for v in PREDEFINED.values() if v[0] == choice]
        if len(matches) == 1:
            keyname, expr, title = matches[0]
        else:
            print("\nInvalid choice; defaulting to SquareRoot (radical).")
            keyname, expr, title = PREDEFINED["5"]

    problem = create_problem_from_expression(expr, title)
    print("\nSolving problem key:", keyname)
    print("Problem:", problem.name)
    return keyname, problem


# ==============================================================================
# PART 4: PDE COMPUTATION (AUTODIFF)
# ==============================================================================

def compute_laplacian(u, x, y):
    # Computes ∇²u = ∂²u/∂x² + ∂²u/∂y² on the NETWORK'S OUTPUT using autodiff.
    # This is different from the SymPy Laplacian — SymPy computed the Laplacian
    # of the true solution symbolically. This computes the Laplacian of whatever
    # the network predicted, numerically, so we can compare the two.
    #
    # requires_grad=True on the input points (set back in sample_interior) is
    # what makes this possible — PyTorch tracked those values through the network
    # so it can differentiate backwards through them now.

    # First derivatives ∂u/∂x and ∂u/∂y
    # grad_outputs=ones_like needed because u is a vector (one value per point),
    # not a scalar — this correctly sums gradients across the batch.
    # create_graph=True keeps the computation graph alive for the second derivative.
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Second derivatives ∂²u/∂x² and ∂²u/∂y²
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    return u_xx + u_yy


# ==============================================================================
# PART 5: LOSS FUNCTIONS
# ==============================================================================
# The loss is essentially two residuals:
#   Interior: how far is the network's Laplacian from f_source? (should be 0)
#   Boundary: how far are the network's predictions from the boundary values? (should be 0)
# When both residuals reach zero, the network has found the correct PDE solution.
# The network never compares to u_true during training — it only uses these physics
# checks. The math guarantees: satisfy both conditions = correct solution.

def compute_interior_loss(model, problem, n_interior):
    x_interior, y_interior = sample_interior(n_interior, device=device)

    # Feed points through the PINN network (the blueprint from Part 1)
    # This is where that structure actually gets used — called thousands of times
    u_pred = model(x_interior, y_interior)

    # Compute Laplacian of the network's prediction
    laplacian_u = compute_laplacian(u_pred, x_interior, y_interior)

    # Get what the Laplacian SHOULD be (auto-derived by SymPy for chosen function)
    f_val = problem.f_source(x_interior, y_interior)

    # Residual = how badly the network is violating the PDE at these points
    residual = laplacian_u - f_val
    return torch.mean(residual ** 2)


def compute_boundary_loss(model, problem, n_per_edge):
    x_boundary, y_boundary = sample_boundary(n_per_edge, device=device)

    # Network's prediction at boundary points
    u_pred_boundary = model(x_boundary, y_boundary)

    # What the network SHOULD predict at those boundary locations
    # (changes based on which function was chosen — e.g. 0 for trig, x²+y² for quadratic)
    g_val = problem.g_boundary(x_boundary, y_boundary)

    return torch.mean((u_pred_boundary - g_val) ** 2)


def compute_total_loss(model, problem, n_interior, n_per_edge, lambda_bc):
    li = compute_interior_loss(model, problem, n_interior)
    lb = compute_boundary_loss(model, problem, n_per_edge)

    # lambda_bc = 10 weights the boundary loss higher — boundary conditions
    # are easier to enforce and the extra weight helps the network learn them quickly
    return li + lambda_bc * lb, li, lb


# ==============================================================================
# PART 6: TRAINING
# ==============================================================================
# This is where the actual learning happens — everything above was just setup.
# The training loop is identical regardless of which problem was chosen.
# That's the payoff of the ProblemConfig abstraction — swap the problem, not the loop.

keyname, problem = choose_problem()

# Create the network from the blueprint defined in Part 1 and send to GPU/CPU
model = PINN().to(device)

# Adam optimizer — decides how to update weights to reduce the loss each step.
# lr (learning rate) controls step size: too big = overshoots, too small = slow.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 5000   # Number of training iterations
n_interior = 1000  # Random interior points per epoch
n_per_edge = 250   # Random boundary points per edge (total = 4×250 = 1000)
lambda_bc = 10.0   # Boundary loss weight

print("\nStarting training...")
print(f"Network parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Training for {n_epochs} epochs")
print("-" * 70)

for epoch in range(1, n_epochs + 1):
    # Step 1: Clear gradients from previous iteration (PyTorch accumulates by default)
    optimizer.zero_grad()

    # Step 2: Sample new random points and measure how wrong the network is
    loss, loss_interior, loss_boundary = compute_total_loss(
        model, problem, n_interior, n_per_edge, lambda_bc
    )

    # Step 3: Backpropagation — figure out which direction to nudge each weight
    loss.backward()

    # Step 4: Actually update the weights based on those directions
    # Repeat 5000 times → network gradually shapes itself into the correct solution
    optimizer.step()

    if epoch % 500 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d} | "
            f"Total: {loss.item():.3e} | "
            f"Interior: {loss_interior.item():.3e} | "
            f"Boundary: {loss_boundary.item():.3e}"
        )

print("-" * 70)
print("Training complete!")


# ==============================================================================
# PART 7: EVALUATION AND VISUALIZATION
# ==============================================================================
# This is the FIRST TIME we compare against the true solution u_true.
# During training the network never saw it — it only used physics checks.
# Now we finally measure how well satisfying those physics checks worked.

def evaluate_accuracy(model, problem, n_test=100):
    # Unlike training (random points), evaluation uses a clean uniform 100×100 grid
    # = 10,000 evenly spaced points for a smooth picture of performance everywhere
    x_test = torch.linspace(0, 1, n_test, device=device)
    y_test = torch.linspace(0, 1, n_test, device=device)
    X, Y = torch.meshgrid(x_test, y_test, indexing="ij")

    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    # torch.no_grad() turns off gradient tracking — not needed for evaluation,
    # saves memory and computation
    with torch.no_grad():
        U_pred_flat = model(X_flat, Y_flat)
    U_pred = U_pred_flat.reshape(n_test, n_test)

    # First time u_true appears — compare network prediction against true answer.
    # u_true comes from ProblemConfig so it's the right answer for whichever
    # function was chosen (different comparison for every function)
    U_true = problem.u_true(X, Y)
    error = U_pred - U_true

    mae = torch.mean(torch.abs(error)).item()
    max_error = torch.max(torch.abs(error)).item()

    # L2 relative error: dividing by norm of U_true makes it a percentage —
    # 0.01 means 1% error regardless of the scale of u
    l2_relative_error = (torch.norm(error) / torch.norm(U_true)).item()

    return {
        "mae": mae,
        "max_error": max_error,
        "l2_relative_error": l2_relative_error,
        "U_pred": U_pred.detach().cpu(),
        "U_true": U_true.detach().cpu(),
        "error": error.detach().cpu(),
        "X": X.detach().cpu(),
        "Y": Y.detach().cpu(),
    }


def print_metrics_table(results):
    print("\n" + "=" * 70)
    print("ACCURACY METRICS")
    print("=" * 70)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 70)
    print(f"{'Mean Absolute Error':<30} {results['mae']:>15.6e}")
    print(f"{'Maximum Absolute Error':<30} {results['max_error']:>15.6e}")
    print(f"{'L2 Relative Error':<30} {results['l2_relative_error'] * 100:>14.4f}%")
    print("=" * 70)


def plot_results(results, problem_label=""):
    # 2×3 grid of plots:
    # Top row: 3D surface plots — True Solution | PINN Prediction | Absolute Error
    # Bottom row: same three things viewed from above as 2D heatmaps
    # If the network did well, true solution and prediction should look nearly identical.
    # The error plot shows exactly where the network struggled most.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X = results["X"].numpy()
    Y = results["Y"].numpy()
    U_true = results["U_true"].numpy()
    U_pred = results["U_pred"].numpy()
    error = np.abs(results["error"].numpy())

    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    surf1 = ax1.plot_surface(X, Y, U_true, cmap="viridis", edgecolor="none",
                             alpha=0.9, linewidth=0, antialiased=True)
    ax1.set_title(f"True Solution ({problem_label})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("u(x,y)")
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    surf2 = ax2.plot_surface(X, Y, U_pred, cmap="viridis", edgecolor="none",
                             alpha=0.9, linewidth=0, antialiased=True)
    ax2.set_title("PINN Prediction", fontsize=12, fontweight="bold")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("u(x,y)")
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    surf3 = ax3.plot_surface(X, Y, error, cmap="hot", edgecolor="none",
                             alpha=0.9, linewidth=0, antialiased=True)
    ax3.set_title("Absolute Error |pred-true|", fontsize=12, fontweight="bold")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("|error|")
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    ax4 = fig.add_subplot(2, 3, 4)
    im1 = ax4.contourf(X, Y, U_true, levels=20, cmap="viridis")
    ax4.set_title("True (Top View)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_aspect("equal")
    fig.colorbar(im1, ax=ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    im2 = ax5.contourf(X, Y, U_pred, levels=20, cmap="viridis")
    ax5.set_title("Prediction (Top View)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("x"); ax5.set_ylabel("y"); ax5.set_aspect("equal")
    fig.colorbar(im2, ax=ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    im3 = ax6.contourf(X, Y, error, levels=20, cmap="hot")
    ax6.set_title("Absolute Error (Top View)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("x"); ax6.set_ylabel("y"); ax6.set_aspect("equal")
    fig.colorbar(im3, ax=ax6)

    plt.tight_layout()
    plt.show()


print("\nEvaluating model on test grid...")
results = evaluate_accuracy(model, problem, n_test=100)
print_metrics_table(results)
plot_results(results, problem_label=keyname)
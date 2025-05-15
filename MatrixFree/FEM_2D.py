import jax, jax.numpy as jnp
from jax import jit
import numpy as np
from jax import lax, debug
from jax.scipy.sparse.linalg import cg
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import time

# Enable double precision if desired
#jax.config.update("jax_enable_x64", True)
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    # Run on single GPU
    DEVICE_ID = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    os.system("clear")
except:
    # Run on CPU
    os.system("clear")
    print("No GPU found.")

#── Geometry & mesh ─────────────────────────────────────────────────
Lx, Ly = 1.0, 1.0       # domain size
Nx, Ny = 128, 128         # # elements in x, y
num_elems = Nx * Ny
num_nodes_x = Nx + 1
num_nodes   = num_nodes_x * (Ny + 1)
dx, dy = Lx / Nx, Ly / Ny

def node_id(i, j):
    return i + j * num_nodes_x

#── Material (plane stress stiffness matrix) ──────────────────────────
E, nu = 210e9, 0.3
C = (E/(1-nu**2)) * jnp.array([
    [1,   nu,      0],
    [nu,  1,       0],
    [0,   0, (1-nu)/2]
])

#── Precompute reference‐element B, detJ, weights for 2×2 Gauss ────────
gp = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
W1 = jnp.array([1.0, 1.0])
GP = jnp.array([(xi, eta) for xi in gp for eta in gp])   # 4 × (ξ,η)
W  = jnp.array([wxi*weta for wxi in W1 for weta in W1])  # 4 weights

@jit
def make_B_detJ():
    # B_all: the 4 strain-displaement matrices B (size 3x8), one for each Gauss point
    # detJ: determinant of the Jacobian (constant for all elements)
    # W: Weights for 2x2 Gaussian quadrature
    # Since all elements are uniform and aligned, we only need to compute the reference quantities once

    # constant Jacobian for uniform quads
    J = jnp.diag(jnp.array([dx/2, dy/2]))
    detJ = jnp.linalg.det(J)
    invJ = jnp.linalg.inv(J)

    def B_at_gp(xi, eta):
        # ∂N/∂(ξ,η) for N1…N4
        dN = jnp.array([
            [-(1-eta), -(1-xi)],
            [ (1-eta), -(1+xi)],
            [ (1+eta),  (1+xi)],
            [-(1+eta),  (1-xi)]
        ]) * 0.25  # shape (4,2)

        # map to ∂N/∂(x,y)
        dN_xy = (invJ @ dN.T).T  # shape (4,2)

        # build B (3×8) from dN_xy
        B = jnp.zeros((3, 8))
        for a in range(4):
            dNx, dNy = dN_xy[a]
            B = B.at[0,2*a   ].set(dNx)
            B = B.at[1,2*a+1 ].set(dNy)
            B = B.at[2,2*a   ].set(dNy)
            B = B.at[2,2*a+1 ].set(dNx)
        return B

    B_all = jnp.stack([B_at_gp(xi, eta) for xi, eta in GP])  # (4,3,8)
    return B_all, detJ, W

B_all, detJ, W = make_B_detJ()

#── Matrix-free operator for 2D elasticity ────────────────────────────
def generate_element_connectivity(Nx, Ny):
    conn = []
    for e in range(Nx * Ny):
        i, j = e % Nx, e // Nx
        nodes = jnp.array([
            node_id(i,   j),
            node_id(i+1, j),
            node_id(i+1, j+1),
            node_id(i,   j+1)
        ])
        conn.append(nodes)
    return jnp.array(conn)  # shape: (num_elems, 4)

conn = generate_element_connectivity(Nx, Ny)  # shape (num_elems, 4)

@jit
def gather_u_e(nodes, u):
    def get_dof(n):
        return lax.dynamic_slice(u, (2 * n,), (2,))
    return jax.vmap(get_dof)(nodes).reshape(-1)

@jit
def compute_element_force(u_e):
    def force_at_gp(B, w):
        eps = B @ u_e
        sig = C @ eps
        return B.T @ sig * detJ * w
    return jnp.sum(jax.vmap(force_at_gp)(B_all, W), axis=0)

@jit
def apply_K2D(u):
    u_e_all = jax.vmap(lambda nodes: gather_u_e(nodes, u))(conn)  # shape: (num_elems, 8)
    f_e_all = jax.vmap(compute_element_force)(u_e_all)            # shape: (num_elems, 8)

    R = jnp.zeros_like(u)

    # Flatten indices for scatter_add
    elem_node_ids = conn.reshape(-1)  # (num_elems * 4,)
    dof_ids = jnp.repeat(2 * elem_node_ids, 2) + jnp.tile(jnp.array([0,1]), num_elems * 4)
    fe_flat = f_e_all.reshape(-1)
    R = R.at[dof_ids].add(fe_flat)

    return R

#── Residual & Jacobian-vector product ────────────────────────────────
@jit
def residual2D(u, f):
    return apply_K2D(u) - f

@jit
def Jv2D(u, v, f):
    return jax.jvp(lambda uu: residual2D(uu, f), (u,), (v,))[1]

#── Newton–CG solver (Dirichlet u=0 at left edge) ──────────────────────
def newton_cg_2D(f, maxit=10, tol=1e-4):
    num_dof = 2 * num_nodes
    # fix all DOFs on left boundary (i=0)
    left_nodes = jnp.array([node_id(0, j) for j in range(Ny+1)])
    free = jnp.setdiff1d(jnp.arange(num_dof), jnp.repeat(2*left_nodes[:,None], 2, axis=1).ravel())

    u = jnp.zeros(num_dof)
    for it in range(maxit):
        R = residual2D(u, f)[free]
        
        def mv(vf):
            # fully inlined, not calling any extra function
            v_full = jnp.zeros_like(u).at[free].set(vf)
            return Jv2D(u, v_full, f)[free]
        
        start_time = time.perf_counter()
        delta, info = cg(jit(mv), -R, tol=1e-4, maxiter=500)
        u = u.at[free].add(delta)
        jax.block_until_ready(u)  # Ensure computation completes
        solver_time = time.perf_counter() - start_time
        print(f"Iteration {it}: Solver time: {solver_time}")

        if jnp.linalg.norm(delta) < tol:
            break

    return u

#── Example: uniaxial tension load on right edge ───────────────────────
f = jnp.zeros(2*num_nodes)
# apply traction of 1e3 N/m on right boundary nodes:
right_nodes = jnp.array([node_id(Nx, j) for j in range(Ny+1)])
for n in right_nodes:
    f = f.at[2*n].add(1e3 * dy)  # x-traction


u_sol = newton_cg_2D(f)

print("Repeating the solver one more time!")
u_sol = newton_cg_2D(f)

# x = np.linspace(0,Lx,num_nodes_x)
# y = np.linspace(0,Ly,num_nodes_x)
# X , Y = np.meshgrid(x,y)
# # Extract UX, UY and reshape to grid
# ux = np.asarray(u_sol[0::2])
# uy = np.asarray(u_sol[1::2])

# UX = ux.reshape((Ny+1, Nx+1))
# UY = uy.reshape((Ny+1, Nx+1))

# # Plot X‐displacement surface
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, UX, cmap='viridis', edgecolor='k', linewidth=0.2)
# ax.set_title('X‐Displacement Field')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('u_x')
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.tight_layout()

# # Plot Y‐displacement surface
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, UY, cmap='plasma', edgecolor='k', linewidth=0.2)
# ax.set_title('Y‐Displacement Field')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('u_y')
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.tight_layout()

# plt.show()
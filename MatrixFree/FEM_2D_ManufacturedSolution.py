import jax, jax.numpy as jnp
from jax import jit
import numpy as np
from jax import lax, debug
from jax.scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import time

# Enable double precision if desired
jax.config.update("jax_enable_x64", True)
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
Nx, Ny = 16, 16         # # elements in x, y
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
    # constant Jacobian for uniform quads
    J = jnp.diag(jnp.array([dx/2, dy/2]))
    detJ = jnp.linalg.det(J)
    invJ = jnp.linalg.inv(J)

    def B_N_at_gp(xi, eta):
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

        # shape function values at this Gauss point
        N = jnp.array([
            0.25 * (1 - xi) * (1 - eta),  # N1
            0.25 * (1 + xi) * (1 - eta),  # N2
            0.25 * (1 + xi) * (1 + eta),  # N3
            0.25 * (1 - xi) * (1 + eta),  # N4
        ])  # shape (4,)

        return B, N

    B_N_list = [B_N_at_gp(xi, eta) for xi, eta in GP]
    B_all = jnp.stack([bn[0] for bn in B_N_list])  # shape (4,3,8)
    N_gp  = jnp.stack([bn[1] for bn in B_N_list])  # shape (4,4)

    return B_all, N_gp, detJ, W

B_all, N_gp, detJ, W = make_B_detJ()

def f_mms(x, y, E=210e9, nu=0.3):
    pi = jnp.pi
    coef = pi**2 * E * (1 - nu) / (1 - nu**2)
    f1 = coef * jnp.sin(pi * x) * jnp.sin(pi * y)
    f2 = coef * jnp.cos(pi * x) * jnp.cos(pi * y)
    return jnp.array([f1, f2])

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
def compute_body_force_vector():
    # Element indices: shape (num_elems, 2)
    elem_ij = jnp.stack(jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij'), axis=-1).reshape(-1, 2)

    # Compute Gauss point coordinates for all elements: shape (num_elems, 4, 2)
    x_gp = (2 * elem_ij[:, 0:1] + 1 + GP[:, 0]) * dx / 2
    y_gp = (2 * elem_ij[:, 1:2] + 1 + GP[:, 1]) * dy / 2

    # Shape (num_elems, 4, 2)
    f_gp = jax.vmap(jax.vmap(f_mms))(x_gp, y_gp)

    # N_gp: shape (4, 4) → reshape to (1, 4, 4) to broadcast over elements
    N_ext = N_gp[None, :, :]  # (1, 4, 4)

    # Weight and detJ
    weight = W * detJ  # (4,)

    # Compute f_ext using einsum for clarity and speed
    # Result: (num_elems, 8)
    # Expand N_gp to act on vector fields (2 DOF per node)
    # N_gp: (4,4) → (4,4,2)
    N_vector = jnp.zeros((4, 4, 2)).at[:, :, 0].set(N_gp).at[:, :, 1].set(N_gp)

    # Reshape f_gp: (num_elems, 4, 2)
    # weight: (4,)
    # Compute: f_ext[e, a, d] = sum_k w_k * N_k[a] * f_gp[e, k, d]
    f_ext = jnp.einsum('k,ekd,kad->ead', weight, f_gp, N_vector)  # shape: (num_elems, 4, 2)

    # Reshape to (num_elems, 8)
    f_ext = f_ext.reshape((num_elems, 8))

    return f_ext  # shape: (num_elems, 8)

@jit
def apply_K2D(u):
    u_e_all = jax.vmap(lambda nodes: gather_u_e(nodes, u))(conn)
    f_int_all = jax.vmap(compute_element_force)(u_e_all)
    f_ext_all = compute_body_force_vector()

    f_total = f_int_all - f_ext_all

    # Flatten for scatter_add
    elem_node_ids = conn.reshape(-1)  # (num_elems * 4,)
    dof_ids = jnp.repeat(2 * elem_node_ids, 2) + jnp.tile(jnp.array([0,1]), num_elems * 4)
    f_flat = f_total.reshape(-1)

    R = jnp.zeros_like(u).at[dof_ids].add(f_flat)
    return R

#── Residual & Jacobian-vector product ────────────────────────────────
@jit
def residual2D(u, f):
    return apply_K2D(u) - f

@jit
def Jv2D(u, v, f):
    return jax.jvp(lambda uu: residual2D(uu, f), (u,), (v,))[1]

#── Newton–CG solver (Dirichlet BCs: fix solution at boundary to manufactured values) ──────────────────────
def newton_cg_2D(f, maxit=1, tol=1e-6):
    num_dof = 2 * num_nodes

    # get all boundary node indices
    boundary_nodes = []
    for j in range(Ny+1):
        for i in range(Nx+1):
            if i == 0 or i == Nx or j == 0 or j == Ny:
                boundary_nodes.append(node_id(i, j))
    boundary_nodes = jnp.array(boundary_nodes)
    all_dofs = jnp.arange(2 * num_nodes)
    boundary_dofs = jnp.ravel(jnp.column_stack((2*boundary_nodes, 2*boundary_nodes+1)))
    free = jnp.setdiff1d(all_dofs, boundary_dofs)

    # initial u set to exact solution at boundary
    u = jnp.zeros(num_dof)
    for j in range(Ny+1):
        for i in range(Nx+1):
            n = node_id(i, j)
            x_n, y_n = i * dx, j * dy
            val_x = jnp.sin(jnp.pi * x_n) * jnp.sin(jnp.pi * y_n)
            val_y = jnp.cos(jnp.pi * x_n) * jnp.cos(jnp.pi * y_n)
            u = u.at[2*n  ].set(val_x)
            u = u.at[2*n+1].set(val_y)

    for _ in range(maxit):
        R = residual2D(u, f)[free]

        def mv(vf):
            # fully inlined, not calling any extra function
            v_full = jnp.zeros_like(u).at[free].set(vf)
            return Jv2D(u, v_full, f)[free]

        start_time = time.perf_counter()
        delta, info = cg(jit(mv), -R, tol=1e-4, maxiter=500)
        solver_time = time.perf_counter() - start_time
        print(solver_time)

        u = u.at[free].add(delta)

        if jnp.linalg.norm(delta) < tol:
            break

    return u

 #── Example: solve manufactured solution via body force ────────────────
f = jnp.zeros(2*num_nodes)
# No external traction; forcing is via manufactured body forces only

u_sol = newton_cg_2D(f)

def u_exact(x, y):
    return jnp.array([
        jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y),
        jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
    ])

x = np.linspace(0,Lx,num_nodes_x)
y = np.linspace(0,Ly,num_nodes_x)
X , Y = np.meshgrid(x,y)
# Extract UX, UY and reshape to grid
ux = np.asarray(u_sol[0::2])
uy = np.asarray(u_sol[1::2])

UX = ux.reshape((Ny+1, Nx+1))
UY = uy.reshape((Ny+1, Nx+1))

Uex = np.zeros_like(UX)
Vex = np.zeros_like(UY)

for j in range(Ny+1):
    for i in range(Nx+1):
        val = u_exact(x[i], y[j])
        Uex[j, i] = val[0]
        Vex[j, i] = val[1]


#── Plot X‐displacement with exact markers ─────────────────────────────
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, UX, cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)
ax.scatter(X, Y, Uex, color='k', s=10, label='Exact', depthshade=False)
ax.set_title('X‐Displacement Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u_x')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()

#── Plot Y‐displacement with exact markers ─────────────────────────────
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, UY, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
ax.scatter(X, Y, Vex, color='k', s=10, label='Exact', depthshade=False)
ax.set_title('Y‐Displacement Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u_y')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()

plt.show()
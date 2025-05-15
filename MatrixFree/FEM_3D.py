import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.sparse.linalg import cg
# from pyevtk.hl import gridToVTK
import time
# import matplotlib.pyplot as plt

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    # Run on single GPU
    DEVICE_ID = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    os.system("clear")
except:
    # Run on CPU
    os.system("clear")
    print("No GPU found.")

# Geometry
Lx = Ly = Lz = 1.0
Nx = Ny = Nz = 32
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

num_nodes = (Nx+1)*(Ny+1)*(Nz+1)
num_elems = Nx * Ny * Nz

# DOF Mapping
def node_id(i, j, k):
    return i + j*(Nx+1) + k*(Nx+1)*(Ny+1)

# Reverse mapping from element index e into its (i,j,k) position in the 3D grid of elements
def element_node_ids(e):
    i = e % Nx          # element index in x    
    j = (e // Nx) % Ny  # element index in y
    k = e // (Nx * Ny)  # element index in z
    base = node_id(i, j, k)

    offsets = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    node_ids = node_id(i + offsets[:, 0], j + offsets[:, 1], k + offsets[:, 2])
    return node_ids  # shape: (8,)

# sol = element_node_ids(14)
# print("Element node IDs for element 1:", sol)
# Material matrix (Voigt)
E, nu = 210e9, 0.3
lam = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
C = jnp.array([
    [lam+2*mu, lam, lam, 0, 0, 0],
    [lam, lam+2*mu, lam, 0, 0, 0],
    [lam, lam, lam+2*mu, 0, 0, 0],
    [0, 0, 0, mu, 0, 0],
    [0, 0, 0, 0, mu, 0],
    [0, 0, 0, 0, 0, mu]
])

# Gauss points and shape functions
gp = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
GP = jnp.array([(x,y,z) for x in gp for y in gp for z in gp])
W = jnp.ones(8)
J = jnp.diag(jnp.array([dx/2, dy/2, dz/2]))
invJ = jnp.linalg.inv(J)
detJ = jnp.linalg.det(J)

def precompute_B():
    B_all = []
    for xi, eta, zeta in GP:
        dN = jnp.array([
            [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
            [ (1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
            [ (1+eta)*(1-zeta),  (1+xi)*(1-zeta), -(1+xi)*(1+eta)],
            [-(1+eta)*(1-zeta),  (1-xi)*(1-zeta), -(1-xi)*(1+eta)],
            [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta),  (1-xi)*(1-eta)],
            [ (1-eta)*(1+zeta), -(1+xi)*(1+zeta),  (1+xi)*(1-eta)],
            [ (1+eta)*(1+zeta),  (1+xi)*(1+zeta),  (1+xi)*(1+eta)],
            [-(1+eta)*(1+zeta),  (1-xi)*(1+zeta),  (1-xi)*(1+eta)],
        ]) * 0.125
        dN_xyz = (invJ @ dN.T).T
        B = jnp.zeros((6, 24))
        for a in range(8):
            dNx, dNy, dNz = dN_xyz[a]
            B = B.at[0, 3*a+0].set(dNx)
            B = B.at[1, 3*a+1].set(dNy)
            B = B.at[2, 3*a+2].set(dNz)
            B = B.at[3, 3*a+0].set(dNy)
            B = B.at[3, 3*a+1].set(dNx)
            B = B.at[4, 3*a+1].set(dNz)
            B = B.at[4, 3*a+2].set(dNy)
            B = B.at[5, 3*a+0].set(dNz)
            B = B.at[5, 3*a+2].set(dNx)
        B_all.append(B)
    return jnp.stack(B_all)

B_all = precompute_B()

# Force application
@jit
def gather_u_e(e, u):
    nodes = element_node_ids(e)
    return u[nodes[:, None]*3 + jnp.arange(3)].reshape(-1)

@jit
def compute_element_force(u_e):
    def gp_force(B, w):
        eps = B @ u_e
        sig = C @ eps
        return B.T @ sig * detJ * w
    return jnp.sum(jax.vmap(gp_force)(B_all, W), axis=0)

@jit
def apply_K3D(u):
    def compute_for_element(e):
        nodes = element_node_ids(e)
        u_e = u[nodes[:, None]*3 + jnp.arange(3)].reshape(-1)
        f_e = compute_element_force(u_e)
        dofs = nodes[:, None] * 3 + jnp.arange(3)
        return dofs.reshape(-1), f_e

    all_dofs, all_forces = jax.vmap(compute_for_element)(jnp.arange(num_elems))
    dofs_flat = all_dofs.flatten()
    forces_flat = all_forces.flatten()

    return jnp.zeros_like(u).at[dofs_flat].add(forces_flat)

@jit
def residual3D(u, f): return apply_K3D(u) - f
@jit
def Jv3D(u, v, f): return jax.jvp(lambda uu: residual3D(uu, f), (u,), (v,))[1]

def newton_cg(f, maxit=2, tol=1e-5):
    num_dof = 3 * num_nodes
    fixed = [node_id(0,j,k) for j in range(Ny+1) for k in range(Nz+1)]
    fixed_dofs = jnp.ravel(jnp.array([[3*n, 3*n+1, 3*n+2] for n in fixed]))
    free = jnp.setdiff1d(jnp.arange(num_dof), fixed_dofs)

    u = jnp.zeros(num_dof)
    for it in range(maxit):
        R = residual3D(u, f)[free]
        # def mv(vf): return Jv3D(u, u.at[free].set(vf), f)[free]
        def mv(vf):
            # make a full-length direction vector that is zero at fixed DOFs
            v_full = jnp.zeros_like(u).at[free].set(vf)
            # Jv3D(u, v_full, f) returns a full-length result; then restrict back to free DOFs
            return Jv3D(u, v_full, f)[free]

        start = time.time()
        delta, _ = cg(jit(mv), -R, tol=tol, maxiter=1000)
        print("CG Time:", time.time() - start)
        u = u.at[free].add(delta)
        #print(f"Iter {it}: ||R|| = {jnp.linalg.norm(R):.3e}")
        if jnp.linalg.norm(R) < tol: break
    return u

# Apply uniaxial traction on x+ face
f = jnp.zeros(3 * num_nodes)
for j in range(Ny+1):
    for k in range(Nz+1):
        n = node_id(Nx, j, k)
        f = f.at[3*n].add(1e6 * dy * dz)

# Solve
Time = []
for _ in range(3):
    start = time.time()
    u_sol = newton_cg(f)
    Time = np.append(Time, time.time() - start)
    print("Solve Time:", Time[-1])

print("Average Solve Time:", np.mean(Time[1:-1]))
# # Export to VTK
# def export_displacement_vtk(X, Y, Z, u, v, w, filename="structured_output"):
#     """
#     Export 3D nodal displacements to VTK using pyevtk.

#     Parameters:
#         X, Y, Z : 3D numpy arrays of nodal coordinates (shape: (nx+1, ny+1, nz+1))
#         Ux, Uy, Uz : 3D numpy arrays of displacement components (same shape as X)
#         filename : base output filename
#     """
#     output_file = f"{filename}"

#     gridToVTK(
#         output_file,
#         np.ascontiguousarray(X),
#         np.ascontiguousarray(Y),
#         np.ascontiguousarray(Z),
#         pointData={
#             "ux": np.ascontiguousarray(u),
#             "uy": np.ascontiguousarray(v),
#             "uz": np.ascontiguousarray(w),
#         }
#     )

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
z = np.linspace(0, Lz, Nz+1)

# meshgrid with axes ordered as (z, y, x)
Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # shape: (Nz+1, Ny+1, Nx+1)

# Displacements must match this shape
u = u_sol[0::3].reshape((Nx+1, Ny+1, Nz+1))  # (Nz+1, Ny+1, Nx+1)
v = u_sol[1::3].reshape((Nx+1, Ny+1, Nz+1))
w = u_sol[2::3].reshape((Nx+1, Ny+1, Nz+1))

# #Export
# export_displacement_vtk(X, Y, Z, u, v, w, filename="structured_output")

# # CPU Time:
# N_DOF   = np.array([17,33,65])**3 * 3
# Time = np.array([1.084,11.488,225.930])

# # GPU Time:
# N_DOF_gpu = np.array([17,33,65,129,257])**3 * 3
# Time_gpu = np.array([0.693,0.870,2.399,18.250,144.887])

# plt.plot(N_DOF,Time,'-o',label='CPU')
# plt.plot(N_DOF_gpu,Time_gpu,'-o',label='GPU')
# plt.plot([60**3,16*60**3],[2,16*2],'--',label='Reference 1')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('DoF')
# plt.ylabel('Time [s]')
# plt.grid()
# plt.legend()
# plt.show()

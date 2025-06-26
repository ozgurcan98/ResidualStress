import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.sparse.linalg import cg
from pyevtk.hl import gridToVTK
import time
# import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
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
Nx = Ny = Nz = 16
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
    for xi, eta, zeta in GP: # 8 Gauss points
        # N_I(xi, eta, zeta) = 1/8*(1+xi*xi_I)*(1+eta*eta_I)*(1+zeta*zeta_I)
        # Jacobian for uniform hexahedra
        # ∂N/∂(ξ,η,ζ) for N1…N8
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
        # map to ∂N/∂(x,y,z) = J^(-1) * ∂N/∂(ξ,η,ζ)
        dN_xyz = (invJ @ dN.T).T

        # Compute B matrix which maps displacement to strain
        # For a single point it is 6 x 3. We have 8 nodes in an element
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

@jit
def f_mms():
    val = -(2 * lam + 4 * mu)
    return jnp.array([val, val, val])

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
    return jnp.sum(jax.vmap(gp_force)(B_all, W), axis=0) # map over 8 entries of GP

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

# Jacobian-vector product in 3D. Jv3D(u, v, f) computes J(u) * v
# where J is the Jacobian of the residual function.
# Jv3D(u, v, f) = dR/du * v = dR/du * (dR/dv) * dv
@jit
def Jv3D(u, v, f): return jax.jvp(lambda uu: residual3D(uu, f), (u,), (v,))[1]

def newton_cg(f, maxit=2, tol=1e-5):

    num_dof = 3 * num_nodes
    # get all boundary node indices & initial u set to exact solution at boundary
    u = jnp.zeros(num_dof)
    boundary_nodes = []
    # for k in range(Nz+1):
    #     for j in range(Ny+1):
    #         for i in range(Nx+1):
    #             if i == 0 or i == Nx or j == 0 or j == Ny or k == 0 or k == Nz:
    #                 nid = node_id(i, j, k)
    #                 boundary_nodes.append(nid)
    #                 x_n, y_n, z_n = i*dx, j*dy, k*dz
    #                 u = u.at[3*nid+0].set(x_n**2)
    #                 u = u.at[3*nid+1].set(y_n**2)
    #                 u = u.at[3*nid+2].set(z_n**2)
                
    # before your Newton solve, build boundary‐DOF indices and values once:
    boundary_nodes = jnp.array(boundary_nodes)
    all_dofs = jnp.arange(3 * num_nodes)
    boundary_dofs = jnp.ravel(jnp.column_stack((3*boundary_nodes, 3*boundary_nodes+1, 3*boundary_nodes+2)))
    free = jnp.setdiff1d(all_dofs, boundary_dofs)

    # 1) build 3D grid of node‐coords
    ii, jj, kk = jnp.meshgrid(
        jnp.arange(Nx+1), 
        jnp.arange(Ny+1), 
        jnp.arange(Nz+1),
        indexing="ij"
    )  # each is shape (Nx+1, Ny+1, Nz+1)

    # 2) flatten and pick the boundary mask
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()
    kk_flat = kk.ravel()
    mask = (ii_flat==0)|(ii_flat==Nx)|(jj_flat==0)|(jj_flat==Ny)|(kk_flat==0)|(kk_flat==Nz)

    # 3) compute the corresponding node IDs and DOFs
    nids = ii_flat + jj_flat*(Nx+1) + kk_flat*(Nx+1)*(Ny+1)
    b_nids = nids[mask]                 # shape (n_bdry,)
    # the 3 DoFs per node:
    dofs_x = 3*b_nids
    dofs_y = dofs_x + 1
    dofs_z = dofs_x + 2

    # 4) compute exact values at those coords
    x_n = ii_flat[mask] * dx
    y_n = jj_flat[mask] * dy
    z_n = kk_flat[mask] * dz
    vals_x = x_n**2
    vals_y = y_n**2
    vals_z = z_n**2

    # now, whenever you need to reset u on the boundary, just do:
    u = u.at[dofs_x].set(vals_x) \
        .at[dofs_y].set(vals_y) \
        .at[dofs_z].set(vals_z)
    
        # assume you have:
    #   ii_flat, jj_flat, kk_flat  = flattened i,j,k coords of every node
    #   mask                        = boolean array selecting boundary nodes
    #   b_nids = ii_flat[mask] + jj_flat[mask]*(Nx+1) + kk_flat[mask]*(Nx+1)*(Ny+1)

    # 1. boundary_nodes
    boundary_nodes = b_nids
    #    this is exactly the list of node indices on ∂Ω

    # 2. all_dofs
    all_dofs = jnp.arange(3 * num_nodes)
    #    a flat list [0,1,2, …, 3*num_nodes-1]

    # 3. boundary_dofs
    #    each node has 3 DOFs: ux,uy,uz → indices 3*n,3*n+1,3*n+2
    boundary_dofs = jnp.concatenate([
        3*boundary_nodes,
        3*boundary_nodes + 1,
        3*boundary_nodes + 2
    ])
    #    shape = (3 * boundary_nodes.size,)

    # 4. free
    free = jnp.setdiff1d(all_dofs, boundary_dofs)
    #    all the DOFs that are *not* Dirichlet‐fixed

    for it in range(maxit):
        R = residual3D(u, f)[free]

        # You must pass only the increment you want multiplied by the Jacobian
        def mv(vf):
            # make a full-length direction vector that is zero at fixed DOFs
            v_full = jnp.zeros_like(u).at[free].set(vf)
            # Jv3D(u, v_full, f) returns a full-length result; then restrict back to free DOFs
            return Jv3D(u, v_full, f)[free]

        start = time.time()
        delta, _ = cg(jit(mv), -R, tol=1E-5, maxiter=1000)

        print("CG Time:", time.time() - start)
        u = u.at[free].add(delta)
        print(f"Iter {it}: ||R|| = {jnp.linalg.norm(R):.3e}")
        if jnp.linalg.norm(R) < tol: break
    return u

# # Assemble RHS from body force
# f_ext = jnp.zeros(3 * num_nodes)
# @jit
# def compute_element_body_force():
#     f_body = f_mms()  # constant (3,)
#     f_e = jnp.zeros(24)
#     for i, (xi, eta, zeta) in enumerate(GP):
#         N = jnp.array([
#             (1 - xi)*(1 - eta)*(1 - zeta),
#             (1 + xi)*(1 - eta)*(1 - zeta),
#             (1 + xi)*(1 + eta)*(1 - zeta),
#             (1 - xi)*(1 + eta)*(1 - zeta),
#             (1 - xi)*(1 - eta)*(1 + zeta),
#             (1 + xi)*(1 - eta)*(1 + zeta),
#             (1 + xi)*(1 + eta)*(1 + zeta),
#             (1 - xi)*(1 + eta)*(1 + zeta)
#         ]) * 0.125  # shape function values at (xi,eta,zeta)
#         for a in range(8):
#             f_e = f_e.at[3*a+0].add(N[a] * f_body[0] * detJ * W[i])
#             f_e = f_e.at[3*a+1].add(N[a] * f_body[1] * detJ * W[i])
#             f_e = f_e.at[3*a+2].add(N[a] * f_body[2] * detJ * W[i])
#     return f_e
# for e in range(num_elems):
#     nodes = element_node_ids(e)
#     dofs = nodes[:, None] * 3 + jnp.arange(3)
#     f_e = compute_element_body_force()
#     f_ext = f_ext.at[dofs.reshape(-1)].add(f_e)

# — your existing constants —
# GP: (8,3) array of Gauss ξ,η,ζ
# W:  (8,)   array of Gauss weights
# detJ: scalar
# num_elems, num_nodes
# element_node_ids: (e→8) mapping
# f_mms(): → (3,) constant body‐force vector

# 1) Precompute the shape‐function matrix at the 8 Gauss points:
#    N_mat[g,p] = N_p(ξ_g,η_g,ζ_g)
N_mat = jnp.stack([
    jnp.array([
        0.125*(1 - xi)*(1 - eta)*(1 - zeta),
        0.125*(1 + xi)*(1 - eta)*(1 - zeta),
        0.125*(1 + xi)*(1 + eta)*(1 - zeta),
        0.125*(1 - xi)*(1 + eta)*(1 - zeta),
        0.125*(1 - xi)*(1 - eta)*(1 + zeta),
        0.125*(1 + xi)*(1 - eta)*(1 + zeta),
        0.125*(1 + xi)*(1 + eta)*(1 + zeta),
        0.125*(1 - xi)*(1 + eta)*(1 + zeta),
    ])
    for (xi,eta,zeta) in GP
])  # shape (8,8)

# 2) Build per‐element body‐force vector f_e (24,)
#    f_gp[g,d] = f_mms()[d] * detJ * W[g]     shape (8,3)
f_body = f_mms()                           # (3,)
f_gp   = f_body[None, :] * (detJ * W)[:,None]

#    f_e_nodes[p,d] = sum_g N_mat[g,p] * f_gp[g,d]   → (8,3)
f_e_nodes = N_mat.T @ f_gp                   # (8,3)

#    flatten to 24-vector (ux,uy,uz per node)
f_e = f_e_nodes.reshape(-1)                  # (24,)

# 3) Vectorize the element loop to get all DOF indices at once
elem_ids   = jnp.arange(num_elems)
elem_nodes = jax.vmap(element_node_ids)(elem_ids)      # (num_elems,8)

#    dofs[e,p,d] = 3*node + d, with d=0,1,2
dofs = elem_nodes[:,:,None] * 3 + jnp.arange(3)        # (num_elems,8,3)
dofs_flat = dofs.reshape(-1)                           # (num_elems*8*3,)

# 4) Scatter‐add into the global f_ext vector in one shot
f_ext = jnp.zeros(3 * num_nodes)
f_ext = f_ext.at[dofs_flat].add(jnp.tile(f_e, num_elems))

# Solve
start = time.time()
u_sol = newton_cg(f_ext)
print(time.time() - start)
    
# Export to VTK
def export_displacement_vtk(X, Y, Z, u, v, w, filename="structured_output"):
    """
    Export 3D nodal displacements to VTK using pyevtk.

    Parameters:
        X, Y, Z : 3D numpy arrays of nodal coordinates (shape: (nx+1, ny+1, nz+1))
        Ux, Uy, Uz : 3D numpy arrays of displacement components (same shape as X)
        filename : base output filename
    """
    output_file = f"{filename}"

    gridToVTK(
        output_file,
        np.ascontiguousarray(X),
        np.ascontiguousarray(Y),
        np.ascontiguousarray(Z),
        pointData={
            "ux": np.ascontiguousarray(u),
            "uy": np.ascontiguousarray(v),
            "uz": np.ascontiguousarray(w),
        }
    )

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
z = np.linspace(0, Lz, Nz+1)

# meshgrid with axes ordered as (z, y, x)
Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # shape: (Nz+1, Ny+1, Nx+1)

# Displacements must match this shape
u = u_sol[0::3].reshape((Nx+1, Ny+1, Nz+1))  # (Nz+1, Ny+1, Nx+1)
v = u_sol[1::3].reshape((Nx+1, Ny+1, Nz+1))
w = u_sol[2::3].reshape((Nx+1, Ny+1, Nz+1))
#Export
export_displacement_vtk(X, Y, Z, u, v, w, filename="structured_output")

# Compute L2 error norm at Gauss points
@jit
def relative_L2(u):
    # per-element gauss-point error
    def elem_error(e):
        # 1) grab nodal dofs for element e → (8,3)
        nodes = element_node_ids(e)
        u_e = u[nodes[:, None]*3 + jnp.arange(3)].reshape(8, 3)

        # 2) element indices i,j,k
        i = e % Nx
        j = (e // Nx) % Ny
        k = e // (Nx * Ny)
        x0, y0, z0 = i*dx, j*dy, k*dz

        # 3) physical gp coordinates: (8,)
        xi, eta, zeta = GP[:,0], GP[:,1], GP[:,2]
        x_gp = x0 + (xi + 1)/2 * dx
        y_gp = y0 + (eta + 1)/2 * dy
        z_gp = z0 + (zeta + 1)/2 * dz

        # 4) evaluate shape‐functions at each gp for each node → N_mat (8×8)
        #    using the ±1 reference‐node signs
        signs = jnp.array([[-1,-1,-1],
                           [ 1,-1,-1],
                           [ 1, 1,-1],
                           [-1, 1,-1],
                           [-1,-1, 1],
                           [ 1,-1, 1],
                           [ 1, 1, 1],
                           [-1, 1, 1]], dtype=jnp.float64)

        # N_mat[k,a] = 1/8*(1+xi[k]*signs[a,0])*(1+eta[k]*signs[a,1])*(1+zeta[k]*signs[a,2])
        N_mat = (1/8) * (
            (1 + xi[:,None]*signs[:,0]) *
            (1 + eta[:,None]*signs[:,1]) *
            (1 + zeta[:,None]*signs[:,2])
        )  # shape (8,8)

        # 5) interpolate numerical u at gp: (8×8)@(8×3) → (8×3)
        u_h = N_mat @ u_e

        # 6) exact solution at gp
        u_ex = jnp.stack([x_gp**2, y_gp**2, z_gp**2], axis=1)  # (8,3)

        # 7) pointwise squared errors
        diff2 = jnp.sum((u_h - u_ex)**2, axis=1)  # (8,)
        ex2   = jnp.sum(u_ex**2, axis=1)          # (8,)

        # 8) weight by detJ·W
        w = W * detJ   # (8,)
        num = jnp.sum(w * diff2)
        den = jnp.sum(w * ex2)
        return num, den

    # vectorize over elements
    nums, dens = jax.vmap(elem_error)(jnp.arange(num_elems))
    # sum over all elements
    total_num = jnp.sum(nums)
    total_den = jnp.sum(dens)

    return jnp.sqrt(total_num / total_den)

# …then after solving…
rel_error = relative_L2(u_sol)
print("Element size", dx)
print("Relative L2 error @ Gauss points:", rel_error)

# Spatial convergence: (64 bit)
Deltax   = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
L2_error = np.array([0.0932505, 0.023294, 0.005823, 0.0014558, 0.0003639, 9.0986e-05, 2.274564e-05])




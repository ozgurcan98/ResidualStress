import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

def apply_K(u, num_elems, L, E, A=1.0):
    """
    Matrix-free action of the stiffness operator K on u.
    1D linear bar, area A, Young's modulus E, length L, ne elements.
    """
    he = L / num_elems
    R = jnp.zeros_like(u)

    def body(e, R):
        i, j = e, e + 1
        q = (E * A / he) * (u[j] - u[i])
        R = R.at[i].add(-q)
        R = R.at[j].add( q)
        return R

    return jax.lax.fori_loop(0, num_elems, body, R)

def residual(u, num_elems, L, E, A, f):
    """ R(u) = K(u) - f """
    return apply_K(u, num_elems, L, E, A) - f

def Jv(u, v, num_elems, L, E, A, f):
    """
    Jacobian–vector product J(u)·v.
    For linear elasticity this is just apply_K(v).
    """
    return jax.jvp(lambda uu: residual(uu, num_elems, L, E, A, f), (u,), (v,))[1]

def newton_cg(num_elems, L, E, A, f, max_newton=10, tol=1e-8):
    """
    Solve K(u)=f with Dirichlet u[0]=0 via Newton iterations,
    using built-in CG for each linear solve.
    """
    num_nodes = num_elems + 1
    free = jnp.arange(1, num_nodes)
    u = jnp.zeros(num_nodes)

    for _ in range(max_newton):
        R = residual(u, num_elems, L, E, A, f)
        Rf = R[free]

        def apply_Jf(vf):
            v_full = jnp.zeros_like(u).at[free].set(vf)
            return Jv(u, v_full, num_elems, L, E, A, f)[free]

        # built-in CG: solve J·delta = -Rf
        delta_f, info = cg(apply_Jf, -Rf, tol=tol, maxiter=500)
        u = u.at[free].add(delta_f)

        if jnp.linalg.norm(Rf) < tol:
            break

    return u

# — Example usage —
ne = 1000        # number of elements
L  = 1.0       # length
E  = 210e9     # Young's modulus
A  = 1.0e-4    # cross-sectional area
# tip load of 1kN at the last node
f  = jnp.zeros(ne+1).at[-1].set(1e3)

u = newton_cg(ne, L, E, A, f)

x = jnp.linspace(0,L,ne+1)
plt.plot(x,u)
plt.show()
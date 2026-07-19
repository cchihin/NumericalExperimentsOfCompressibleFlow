import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Sod exact Riemann solver (γ = 1.4 typical for air)
# ---------------------------------------------------------------

gamma = 5/3

# Initial left/right states
rhoL, uL, pL = 1.0, 0.0, (gamma - 1.0) * 0.1
rhoR, uR, pR = 1e-3, 0.0, (gamma - 1.0) * 1.e-10

# Sound speed
def sound_speed(rho, p):
    return np.sqrt(gamma * p / rho)

cL = sound_speed(rhoL, pL)
cR = sound_speed(rhoR, pR)

# Functions f_L(p) and f_R(p) from Toro
def f_shock(p, pk, rho_k):
    A = 2.0 / ((gamma + 1) * rho_k)
    B = (gamma - 1) / (gamma + 1) * pk
    return (p - pk) * np.sqrt(A / (p + B))

def f_rarefaction(p, pk, ck):
    return (2 * ck / (gamma - 1)) * ((p / pk)**((gamma - 1) / (2 * gamma)) - 1)

def f_k(p, pk, rho_k, ck):
    if p > pk:
        return f_shock(p, pk, rho_k)
    else:
        return f_rarefaction(p, pk, ck)

# Solve for p* using bisection (safe and robust)
def solve_p_star():
    # p_min, p_max = 0.0001, 10.0
    p_min = 1e-14
    p_max = pL 
    for _ in range(200):
        pm = 0.5*(p_min + p_max)
        f = f_k(pm, pL, rhoL, cL) + f_k(pm, pR, rhoR, cR) + (uR - uL)
        if f > 0:
            p_max = pm
        else:
            p_min = pm
    return 0.5*(p_min + p_max)

p_star = solve_p_star()

# Compute u*
u_star = 0.5*(uL + uR) + 0.5*(f_k(p_star, pR, rhoR, cR) - f_k(p_star, pL, rhoL, cL))

# Star-region densities
def rho_star(p_star, pk, rho_k):
    if p_star > pk:     # shock
        return rho_k * ( (p_star/pk + (gamma - 1)/(gamma + 1)) /
        ( (gamma - 1)/(gamma + 1)*(p_star/pk) + 1 ) )
    else:               # rarefaction
        return rho_k * (p_star/pk)**(1/gamma)

rhoL_star = rho_star(p_star, pL, rhoL)
rhoR_star = rho_star(p_star, pR, rhoR)

# Rarefaction tail sound speed
c_star_L = sound_speed(rhoL_star, p_star)

# Shock speed (right-moving shock)
S = uR + cR * np.sqrt(
(gamma + 1)/(2 * gamma) * (p_star/pR) + (gamma - 1)/(2 * gamma)
)

# Rarefaction fan speeds
xi_head = uL - cL
xi_tail = u_star - c_star_L

# ---------------------------------------------------------------
# Evaluating solution at any (x,t)
# ---------------------------------------------------------------

def leblanc_exact(x, x0, t):
    xi = (x - x0) / t
    rho = np.zeros_like(x)
    u   = np.zeros_like(x)
    p   = np.zeros_like(x)
    E   = np.zeros_like(x)

    for i, s in enumerate(xi):
        # Region 1: Left state
        if s <= xi_head:
            rho[i], u[i], p[i] = rhoL, uL, pL

        # Region 2: Inside rarefaction fan
        elif xi_head < s <= xi_tail:
            u[i] = (2/(gamma+1)) * (cL + s)
            c = cL - (gamma - 1)/2 * u[i]
            rho[i] = rhoL * (c/cL)**(2/(gamma - 1))
            p[i] = pL * (c/cL)**(2*gamma/(gamma - 1))

        # Region 3: Left star state
        elif xi_tail < s <= u_star:
            rho[i], u[i], p[i] = rhoL_star, u_star, p_star

        # Region 4: Right star state
        elif u_star < s <= S:
            rho[i], u[i], p[i] = rhoR_star, u_star, p_star

        # Region 5: Right state
        else:
            rho[i], u[i], p[i] = rhoR, uR, pR
    
        E[i] = p[i]/(gamma - 1) + 0.5 * rho[i] * u[i]**2

    return rho, u, p, E

# ---------------------------------------------------------------
# Plotting example
# ---------------------------------------------------------------

if __name__ == "__main__":

    print(__name__)
    x0 = 3
    x = np.linspace(0, 10, 1000)
    t = 10*2/3  # choose any t > 0

    rho, u_vel, p, E = leblanc_exact(x, x0, t)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0,0].plot(x, rho)
    axs[0,0].set_ylabel("Density")

    axs[0,1].plot(x, u_vel)
    axs[0,1].set_ylabel("Velocity")

    axs[1,0].plot(x, p)
    axs[1,0].set_ylabel("Pressure")
    axs[1,0].set_xlabel("x")

    axs[1,1].plot(x, E)
    axs[1,1].set_ylabel("Pressure")
    axs[1,1].set_xlabel("x")

    plt.tight_layout()
    plt.show()


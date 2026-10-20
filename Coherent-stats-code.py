import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.special import factorial
import matplotlib.animation as animation


def phi(x,n,m,omega,hbar):
    H_n = hermite(n)
    return ((1/np.sqrt(2**n * factorial(n))) * 
    (m*omega/(np.pi*hbar))**0.25 * np.exp(-m*omega*x**2/(2*hbar)) 
    * H_n(np.sqrt(m*omega/hbar)*x))

def psi(x,m,N,omega,hbar, alpha):
    return np.exp(-np.absolute(alpha)**2/2) * np.sum([alpha**k / np.sqrt(factorial(k)) * phi(x,k,m,omega,hbar) for k in range(N)], axis=0)

#Vérification 

def valeur_moyenne_X(m, N, omega, hbar, alpha):
    return np.sqrt(2*hbar/(m*omega)) * np.real(alpha)

def valeur_moyenne_X_numerique(x, psi_t):
    return np.trapezoid(x * np.absolute(psi_t)**2, x)

x = np.linspace(-5, 5, 1000)
m = 1.0
omega = 1.0
hbar = 1.0
alpha = 1.0
N = 10
    
print(valeur_moyenne_X(m, N, omega, hbar, alpha))

plt.figure(figsize=(10, 6))
plt.plot(x, np.absolute(psi(x, m, N, omega, hbar, alpha))**2, label='Coherent State', color='blue')
plt.title('Coherent State Wavefunction')
plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.grid()
plt.legend()
plt.show()

#Evolution temporel 

def psi_time(x,m,omega,hbar, alpha, N ,t):
    #E_n = hbar * omega * (n + 0.5)
    return np.exp(-np.absolute(alpha)**2/2) * np.sum([alpha**k / np.sqrt(factorial(k))*np.exp(-1j* hbar * omega * (k + 0.5)*t/hbar) * phi(x,k,m,omega,hbar) for k in range(N)], axis=0)




plt.figure(figsize=(10, 6))
t_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
for t in t_values:
    psi_t = np.absolute(psi_time(x,m,omega,hbar, alpha, N ,t))**2
    plt.plot(x, np.absolute(psi_t)**2, label=f't={t:.2f}')
plt.title('Time Evolution of Coherent State Wavefunction')
plt.xlabel('x')
plt.ylabel('ψ(x,t)')
plt.grid()
plt.legend()
plt.show()




for t in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
    psi_t = psi_time(x,m,omega,hbar, alpha, N ,t)
    x_num = valeur_moyenne_X_numerique(x, psi_t)
    x_analytic = valeur_moyenne_X(m, N, omega, hbar, alpha) * np.cos(omega*t) 
    print(f"t={t}: numérique={x_num:.4f}, analytique={x_analytic:.4f}")



#Animation

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='blue', label='|ψ(x,t)|²')
point, = ax.plot([], [], 'ro', markersize=10, label='classical position ⟨X⟩(t)')

ax.set_xlim(-5, 5)
ax.set_ylim(0, 1)  
ax.set_xlabel('x')
ax.set_ylabel('|ψ(x,t)|²')
ax.legend()
ax.grid()

def init():  # initialize the plot
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(frame): 
    t = frame * dt 
    psi_t = psi_time(x,m,omega,hbar, alpha, N ,t)
    density = np.absolute(psi_t)**2
    line.set_data(x, density)
    
    x_classique = valeur_moyenne_X(m, N, omega, hbar, alpha) * np.cos(omega*t)
    point.set_data([x_classique], [0]) 
    
    ax.set_title(f'Time t = {t:.2f}')
    return line, point

dt = 0.05
n_frames = 200

ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=50)
ani.save('coherent_state_animation.gif', writer=animation.PillowWriter(fps=20))
plt.show()


def psi_non_coherent(x, sigma, x0=0):
    return (1/(2*np.pi*sigma**2))**0.25 * np.exp(-(x-x0)**2/(4*sigma**2))

#decomposition on |n> basis

def coefficients_projection(x, psi_initial, N, m, omega, hbar):
    coeffs = []
    for n in range(N):
        phi_n = phi(x, n, m, omega, hbar)
        c_n = np.trapezoid(np.conj(phi_n) * psi_initial, x)
        coeffs.append(c_n)
    return np.array(coeffs)

def general_evolution(x, coeffs, m, omega, hbar, t):
    result = np.zeros_like(x, dtype=complex)
    for n, c_n in enumerate(coeffs):
        E_n = hbar*omega*(n+0.5)
        result += c_n * np.exp(-1j*E_n*t/hbar) * phi(x, n, m, omega, hbar)
    return result


plt.figure(figsize=(10, 6))
sigma = 0.2
psi_initial = psi_non_coherent(x, sigma)
plt.plot(x, np.absolute(psi_initial)**2, label='Non-coherent State', color='purple')
coeffs = coefficients_projection(x, psi_initial, N, m, omega, hbar)

#Evolution temporelle

t_values = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
for t in t_values:
    psi_t = general_evolution(x, coeffs, m, omega, hbar, t)
    plt.plot(x, np.absolute(psi_t)**2, label=f't={t:.2f}', linestyle='--')
plt.title('Time Evolution of Non-Coherent State Wavefunction')
plt.xlabel('x')
plt.ylabel('|ψ(x,t)|²')
plt.grid()
plt.legend()
plt.show()

#étalement du paquet d'onde 


#ΔX(t)·ΔP(t) numérique

def delta_X_numerique(x, psi_t):
    x_mean = np.trapezoid(x*np.absolute(psi_t)**2, x)
    x2_mean = np.trapezoid(x**2*np.absolute(psi_t)**2, x)
    return np.sqrt(x2_mean - x_mean**2)


def delta_P_numerique(x, psi_t, hbar):
    dx = x[1] - x[0]
    
    dpsi_dx = np.gradient(psi_t, dx)
    d2psi_dx2 = np.gradient(dpsi_dx, dx)
    
    p_mean = np.trapezoid(np.conj(psi_t) * (-1j*hbar) * dpsi_dx, x).real
    p2_mean = np.trapezoid(np.conj(psi_t) * (-hbar**2) * d2psi_dx2, x).real
    
    return np.sqrt(p2_mean - p_mean**2)



plt.figure(figsize=(10, 6))
delta_X_values = []
delta_P_values = []
for t in np.linspace(0, 2*np.pi, 100):
    psi_t = psi_time(x,m,omega,hbar, alpha, N ,t)
    delta_X = delta_X_numerique(x, psi_t)
    delta_P = delta_P_numerique(x, psi_t, hbar)
    delta_X_values.append(delta_X)
    delta_P_values.append(delta_P)
    psi_non_coherent_t = general_evolution(x, coeffs, m, omega, hbar, t)

plt.plot(np.linspace(0, 2*np.pi, 100), np.array(delta_X_values)*np.array(delta_P_values), label='ΔX(t)·ΔP(t)', color='blue')
plt.axhline(y=hbar/2, color='red', linestyle='--', label='Heisenberg Limit (ħ/2)')
plt.title('Uncertainty Product ΔX(t)·ΔP(t) Over Time')
plt.xlabel('Time t')
plt.ylabel('ΔX(t)·ΔP(t)')
plt.grid()
plt.legend()
plt.show()
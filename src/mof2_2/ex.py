import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt


def ex2():
    base = ex(alpha=1, N=300)
    alpha_max = base.m * base.dx**2
    alphas = [
        0.5 * alpha_max,
        0.8 * alpha_max,
        0.95 * alpha_max,
        0.999 * alpha_max,
        1.001 * alpha_max,
    ]
    iters = []
    Es = []
    for alpha in alphas:
        new_base = ex(alpha=alpha, N=300)
        result = new_base.solve()
        print(
            f"Ex2: alpha: {alpha}; Energy: {ex.au_to_meV(result['energy'])} meV; delta: {result['delta']}; Number of iterations: {result['iterations']}"
        )
        iters.append(result["iterations"])
        Es.append(result['energy'])

    fig, ax = plt.subplots()

    ax.plot(np.array(alphas) / alpha_max, iters)
    ax.set_xlabel("$\\alpha$ [$\\% \\alpha_{max}$]")
    ax.set_ylabel("Liczba iteracji")

    fig.savefig("ex2_alpha.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(np.array(alphas) / alpha_max, ex.au_to_meV(np.array(Es)))
    ax.set_xlabel("$\\alpha$ [$\\% \\alpha_{max}$]")
    ax.set_ylabel("Energia [meV]")

    fig.savefig("ex2_E.svg", format="svg")
    plt.close(fig)


def ex3(nu=0):
    base = ex(alpha=1, N=300, nu=nu)
    alpha = 0.999 * base.m * base.dx**2
    print("Ex4: solving psi1")
    psi1 = ex(alpha=alpha, N=300, nu=nu)
    result1 = psi1.solve()
    print(
        f"Ex4: Energy 1: {ex.au_to_meV(result1['energy'])} meV; delta: {result1["delta"]}; Number of iterations: {result1['iterations']}"
    )

    print("Ex4: solving psi2")
    psi2 = ex(alpha=alpha, N=300, nu=nu)
    result2 = psi2.solve(psis=[psi1])
    print(
        f"Ex4: Energy 2: {ex.au_to_meV(result2['energy'])} meV; delta: {result2["delta"]}; Number of iterations: {result2['iterations']}"
    )

    fig, ax = plt.subplots()
    ax.plot(psi1.x*psi1.A0, psi1.psi, label="$\\Psi_1$")
    ax.plot(psi2.x*psi2.A0, psi2.psi, label="$\\Psi_2$")
    ax.set_xlabel("Położenie [m]")
    ax.set_ylabel("Funkcja falowa $\\Psi$")
    ax.legend()
    ax.grid()

    fig.savefig(f"ex3_nu_{nu}.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(ex.au_to_meV(result1["history"]), label="Zbierzność $E_1$")
    ax.plot(ex.au_to_meV(result2["history"]), label="Zbierzność $E_2$")
    ax.grid()
    ax.set_yscale("log")
    ax.set_xlabel("iteracja")
    ax.set_ylabel("$\\Delta$ E [log meV]")
    ax.legend()

    fig.savefig(f"ex3_zbierznosc_nu_{nu}.svg", format="svg")
    plt.close(fig)


def more_En(N=7):
    psis = []
    Es = []
    base = ex(alpha=1, N=300)
    alpha = 0.999 * base.m * base.dx**2
    for i in range(N):
        new_psi = ex(alpha=alpha, N=300)
        result = new_psi.solve(tolerance=1e-9, psis=psis)
        print(
            f"Ex*: Calculated Energy level {i} with Energy={ex.au_to_meV(result['energy'])} meV, Number of iterations={result['iterations']}"
        )
        psis.append(new_psi)
        Es.append(result['energy'])

    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(psis[i].x, psis[i].psi, label=f"$\\Psi_{i}$")
    
    ax.legend()
    ax.set_xlabel("Położenie [a.u.]")
    ax.set_ylabel("$\\Psi$")
    ax.grid()

    fig.savefig("exX_psis.svg", format='svg')
    plt.close(fig)


class ex:
    def __init__(self, alpha, L=200e-9, m=0.067 * const.m_e, nu=0, N=100):
        self.A0 = const.physical_constants["Bohr radius"][0]
        self.alpha = alpha
        self.N = N
        self.nu = nu
        self.m = m
        self.L = L / self.A0  # in ab
        self.m = m / const.m_e
        self.dx = self.L / N  # in ab
        self.__init_grid()

    def __init_grid(self):
        self.psi = np.zeros(self.N + 1)
        self.psi[1 : self.N] = np.random.rand(self.N - 1)
        self.x = np.linspace(0, self.N * self.dx, self.N + 1)

    def V(self, x):
        a = 10e-9 / self.A0
        return (
            -1
            / (2 * self.m * a**2)
            * self.nu
            * (self.nu + 1)
            / np.cosh((x - self.L / 2) / a) ** 2
        )

    def Hamiltonian(self):
        """
        Hamiltonian returns H * \\Psi
        """
        return (
            -(
                self.psi[2 : self.N + 1]
                + self.psi[0 : self.N - 1]
                - 2 * self.psi[1 : self.N]
            )
            / (2 * self.m * self.dx**2)
            + self.V(self.x[1 : self.N]) * self.psi[1 : self.N]
        )

    def iterate_psi(self):
        """
        Do one psi iteration.
        """
        H = self.Hamiltonian()
        self.psi[1 : self.N]
        self.psi[1 : self.N] = self.psi[1 : self.N] - self.alpha * H
        # normalize
        self.normalize()

    def normalize(self):
        C = np.sqrt(self.dx * (self.psi**2).sum())
        self.psi /= C

    def Energy(self):
        return (self.psi[1 : self.N] * self.Hamiltonian() * self.dx).sum()

    @staticmethod
    def meV_to_au(meV):
        hartree_eV = const.physical_constants["Hartree energy in eV"][0]
        return meV * 1e-3 / hartree_eV
    @staticmethod
    def au_to_meV(au):
        hartree_eV = const.physical_constants["Hartree energy in eV"][0]
        return au * hartree_eV * 1e3

    def solve(self, tolerance=1e-6, psis=[]):
        """
        solves for Energy. Exits when energy delta is less than the tolerance.
        :param tolerance: tolerance in meV
        """
        tolerance_au = self.meV_to_au(tolerance)
        oldE = 0
        deltas = []
        while True:
            self.iterate_psi()

            for psi in psis:
                c = (psi.psi * self.psi * self.dx).sum()
                self.psi -= c * psi.psi
                self.normalize()

            newE = self.Energy()
            delta = abs(newE - oldE)
            oldE = newE
            deltas.append(abs(delta))
            if delta < tolerance_au:
                return {
                    "energy": oldE,
                    "delta": delta,
                    "history": np.array(deltas),
                    "iterations": len(deltas),
                }

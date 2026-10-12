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
        alpha_max,
    ]
    for alpha in alphas:
        new_base = ex(alpha=alpha, N=300)
        print("---")
        print(new_base.solve())

def ex3(nu=0):
    base = ex(alpha=1, N=300, nu=nu)
    alpha= .999 * base.m * base.dx**2
    print("Ex3: solving psi1")
    psi1 = ex(alpha=alpha, N=300, nu=nu)
    print(psi1.solve())
    print("Ex3: solving psi2")
    psi2 = ex(alpha=alpha, N=300, nu=nu)
    print(psi2.solve2(psi1))

    fig, ax = plt.subplots()
    ax.plot(psi1.x, psi1.psi, label="$\\Psi_1$")
    ax.plot(psi2.x, psi2.psi, label="$\\Psi_2$")
    ax.set_xlabel("Położenie [a.u.]")
    ax.set_ylabel("Funkcja falowa $\\Psi$")
    ax.legend()
    ax.grid()

    fig.savefig(f"ex3_nu_{nu}.svg", format='svg')
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

    def solve(self, tolerance=1e-6):
        """
        solves for Energy. Exits when energy delta is less than the tolerance.
        :param tolerance: tolerance in meV
        """
        tolerance_au = self.meV_to_au(tolerance)
        oldE = 0
        while True:
            self.iterate_psi()
            newE = self.Energy()
            delta = abs(newE - oldE)
            oldE = newE
            if delta < tolerance_au:
                return {
                    "energy": oldE,
                    "delta": delta,
                }

    def solve2(self, psi1, tolerance=1e-6):
        """
        solve2 calculates \\Psi_2
        :param psi1: calcualted \\Psi_1 value
        """
        tolerance_au = self.meV_to_au(tolerance)
        oldE = 0
        while True:
            self.iterate_psi()
            c1 = (psi1.psi * self.psi * self.dx).sum()
            self.psi -= c1 * psi1.psi
            self.normalize()
            newE = self.Energy()
            delta = abs(newE - oldE)
            oldE = newE
            if delta < tolerance_au:
                return {
                    "energy": oldE,
                    "delta": delta,
                }

import scipy.constants as const
import numpy as np


def ex2():
    base = ex(alpha=1, N=100)
    alpha_max = base.m * base.dx**2
    alphas = [0.5 * alpha_max, 0.8 * alpha_max, 0.95 * alpha_max, alpha_max]
    for alpha in alphas:
        new_base = ex(alpha=alpha, N=100)
        print("---")
        print(new_base.solve())


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
        Hamiltonian returns H * \Psi
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
        C = np.sqrt(self.dx * (self.psi**2).sum())
        self.psi /= C

    def Energy(self):
        return (
            self.psi[1 : self.N] * self.Hamiltonian() * self.dx
        ).sum()

    def solve(self, tolerance=1e-6):
        """
        solves for Energy. Exits when energy delta is less than the tolerance.
        :param tolerance: tolerance in meV
        """
        hartree_eV = const.physical_constants["Hartree energy in eV"][0]
        tolerance_au = tolerance * 1e-3 / hartree_eV
        oldE = 0
        while True:
            self.iterate_psi()
            newE = self.Energy()
            delta = abs(newE - oldE)
            print(f"Solver: old energy: {oldE}; new energy: {newE}; delta: {delta}")
            oldE = newE
            if delta < tolerance_au:
                return {
                    "energy": oldE,
                    "delta": delta,
                }

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from scipy.interpolate import UnivariateSpline


def initialize_lattice(L):
    theta = np.random.uniform(0, 2 * np.pi, (L, L, L))
    spins = np.stack((np.cos(theta), np.sin(theta)), axis=-1)  # Shape: (L, L, L, 2)
    return spins

def wolff_update(spins, J, T):
    beta = 1.0 / T
    L = spins.shape[0]

    # Choose a random reflection axis
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.array([np.cos(phi), np.sin(phi)])  # Unit vector in x-y plane

    # Choose a random seed spin
    i0, j0, k0 = np.random.randint(0, L, 3)
    S_i0 = spins[i0, j0, k0]

    # Reflect the seed spin
    S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
    spins[i0, j0, k0] = S_i0_new

    # Initialize cluster
    cluster = set()
    cluster.add((i0, j0, k0))

    # Use a stack for depth-first search
    stack = deque()
    stack.append((i0, j0, k0))

    # Keep track of flipped spins
    flipped = np.zeros(spins.shape[:3], dtype=bool)
    flipped[i0, j0, k0] = True

    while stack:
        i, j, k = stack.pop()
        S_i = spins[i, j, k]

        # Neighbor indices with periodic boundary conditions
        neighbors = [
            ((i + 1) % L, j, k),
            ((i - 1) % L, j, k),
            (i, (j + 1) % L, k),
            (i, (j - 1) % L, k),
            (i, j, (k + 1) % L),
            (i, j, (k - 1) % L)
        ]

        for ni, nj, nk in neighbors:
            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                delta = 2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect spin S_j
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.add((ni, nj, nk))
                    stack.append((ni, nj, nk))

    return len(cluster)

def wolff_update_new(spins, J, T):
    beta = 1.0 / T
    L = spins.shape[0]

    # Choose a random reflection axis
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.array([np.cos(phi), np.sin(phi)])  # Unit vector in x-y plane

    # Choose a random seed spin
    i0, j0, k0 = np.random.randint(0, L, 3)
    S_i0 = spins[i0, j0, k0]

    # Reflect the seed spin
    S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
    spins[i0, j0, k0] = S_i0_new

    # Initialize cluster
    cluster = set()
    cluster.add((i0, j0, k0))

    # Use a stack for depth-first search
    stack = deque()
    stack.append((i0, j0, k0))

    # Keep track of flipped spins
    flipped = np.zeros(spins.shape[:3], dtype=bool)
    flipped[i0, j0, k0] = True

    while stack:
        i, j, k = stack.pop()
        S_i = spins[i, j, k]

        # Neighbor indices with periodic boundary conditions
        neighbors = [
            ((i + 1) % L, j, k),
            ((i - 1) % L, j, k),
            (i, (j + 1) % L, k),
            (i, (j - 1) % L, k),
            (i, j, (k + 1) % L),
            (i, j, (k - 1) % L)
        ]

        for ni, nj, nk in neighbors:
            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                # Corrected delta with negative sign
                delta = -2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect spin S_j
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.add((ni, nj, nk))
                    stack.append((ni, nj, nk))

    return len(cluster)

def wolff_update_with_estimator(spins, J, T):
    beta = 1.0 / T
    L = spins.shape[0]

    # Choose a random reflection axis (unit vector in x-y plane)
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.array([np.cos(phi), np.sin(phi)])  # Reflection axis

    # Choose a random seed spin
    i0, j0, k0 = np.random.randint(0, L, 3)
    S_i0 = spins[i0, j0, k0]

    # Reflect the seed spin
    S_i0_new = S_i0 - 2 * np.dot(S_i0, r) * r
    spins[i0, j0, k0] = S_i0_new

    # Initialize cluster
    cluster = [(i0, j0, k0)]
    flipped = np.zeros((L, L, L), dtype=bool)
    flipped[i0, j0, k0] = True

    # Store positions of spins in the cluster
    cluster_positions = [(i0, j0, k0)]

    # Cluster growth
    while cluster:
        i, j, k = cluster.pop()
        S_i = spins[i, j, k]

        # Neighbor indices with periodic boundary conditions
        neighbors = [
            ((i + 1) % L, j, k),
            ((i - 1) % L, j, k),
            (i, (j + 1) % L, k),
            (i, (j - 1) % L, k),
            (i, j, (k + 1) % L),
            (i, j, (k - 1) % L)
        ]

        for ni, nj, nk in neighbors:
            if not flipped[ni, nj, nk]:
                S_j = spins[ni, nj, nk]
                delta = 2 * beta * J * np.dot(S_i, r) * np.dot(S_j, r)
                p_add = 1 - np.exp(min(0, delta))

                if np.random.rand() < p_add:
                    # Reflect spin S_j
                    S_j_new = S_j - 2 * np.dot(S_j, r) * r
                    spins[ni, nj, nk] = S_j_new
                    flipped[ni, nj, nk] = True
                    cluster.append((ni, nj, nk))
                    cluster_positions.append((ni, nj, nk))

    cluster_size = np.sum(flipped)

    cluster_Sq, q_vectors = compute_improved_structure_factor(cluster_positions, L)

    return cluster_size, cluster_Sq, q_vectors

def compute_improved_structure_factor(cluster_positions, L):
    cluster_size = len(cluster_positions)

    q_vectors = [
        np.array([2 * np.pi / L, 0, 0]),  # Small q in x-direction
        np.array([0, 2 * np.pi / L, 0]),  # Small q in y-direction
        np.array([0, 0, 2 * np.pi / L])  # Small q in z-direction
    ]

    cluster_Sq = np.zeros(len(q_vectors))

    # Convert positions to numpy array
    positions = np.array(cluster_positions)  # Shape: (cluster_size, 3)

    # Compute sum over cluster positions
    for idx, q in enumerate(q_vectors):
        qr = np.dot(positions, q)  # Shape: (cluster_size,)
        sum_exp = np.sum(np.exp(1j * qr))
        cluster_Sq[idx] = (np.abs(sum_exp) ** 2) / (3 * cluster_size)

    return cluster_Sq, q_vectors

def compute_correlation_length_from_Sq(structure_factors, q_vectors, L):
    d = 3  # Spatial dimensionality
    S_q = np.array(structure_factors)  # Structure factors
    q_values = np.array(q_vectors)  # Magnitudes of q vectors

    # Zeroth moment (mu_0): Sum of S(q)
    mu_0 = np.sum(S_q)

    # Second moment (mu_2): Sum of |q|^2 * S(q)
    mu_2 = np.sum((sum(np.abs(q_values)) ** 2) * S_q)

    # Avoid division by zero or invalid results
    if mu_0 <= 0 or mu_2 <= 0:
        return np.inf  # Infinite correlation length if moments are invalid

    # Compute correlation length
    xi = np.sqrt(mu_2 / (2 * d * mu_0))
    return xi

def compute_autocorrelation(data):
    """
    Compute the autocorrelation function of a 1D array.

    Parameters:
        data (array-like): Time series data.

    Returns:
        autocorr (ndarray): Autocorrelation function.
    """
    data = np.asarray(data)
    n = len(data)
    data_mean = np.mean(data)
    data_var = np.var(data)

    autocorr = np.correlate(data - data_mean, data - data_mean, mode='full')
    autocorr = autocorr[n - 1:] / (data_var * n)

    return autocorr

def estimate_autocorrelation_time(autocorr):
    """
    Estimate the integrated autocorrelation time.

    Parameters:
        autocorr (ndarray): Autocorrelation function.

    Returns:
        tau_int (float): Integrated autocorrelation time.
    """
    # Integrated autocorrelation time
    # Sum until the autocorrelation function drops below zero
    positive_autocorr = autocorr[autocorr > 0]
    tau_int = 0.5 + np.sum(positive_autocorr[1:])  # Skip the first term (t=0)

    return tau_int

def compute_susceptibility_with_estimator(magnetizations, V, T):
    M_abs = np.array(magnetizations) * V  # Total magnetization
    M_abs_mean = np.mean(M_abs)
    M2_mean = np.mean(M_abs ** 2)
    chi = (M2_mean - M_abs_mean ** 2) / (V * T)

    N = len(M_abs)
    numerator = M_abs ** 2 - M_abs_mean ** 2
    sigma_numerator = np.std(numerator, ddof=1)
    error_chi = sigma_numerator / (np.sqrt(N) * V * T)

    return chi, error_chi

def compute_F(spins, L):
    """
    Compute F = \hat{G}(k) at |k| = 2π/L.
    """
    V = L ** 3
    # Smallest non-zero momentum vectors
    q_vectors = [
        np.array([2 * np.pi / L, 0, 0]),
        np.array([0, 2 * np.pi / L, 0]),
        np.array([0, 0, 2 * np.pi / L])
    ]

    # Initialize F
    F_values = []

    # Convert spins to complex numbers
    spin_complex = spins[..., 0] + 1j * spins[..., 1]  # Shape: (L, L, L)

    # Coordinates
    x = np.arange(L)
    y = np.arange(L)
    z = np.arange(L)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    positions = np.stack([X, Y, Z], axis=-1)  # Shape: (L, L, L, 3)

    for q in q_vectors:
        # Compute phase factors
        r_dot_q = np.tensordot(positions, q, axes=([3], [0]))  # Shape: (L, L, L)
        phase = np.exp(-1j * r_dot_q)  # Shape: (L, L, L)

        # Compute sum over spins
        sum_sq = np.sum(spin_complex * phase)
        F = np.abs(sum_sq) ** 2 / V
        F_values.append(F)

    # Average over directions
    F_mean = np.mean(F_values)
    return F_mean

def compute_xi_second_moment(chi, F, L):
    pi_over_L = np.pi / L
    sin_term = np.sin(pi_over_L)
    denominator = 4 * sin_term ** 2
    ratio = chi / F
    xi_2nd = np.sqrt(np.abs(ratio - 1) / denominator)
    return xi_2nd

def compute_magnetization(spins):
    total_spin = np.sum(spins, axis=(0, 1, 2))
    M = np.linalg.norm(total_spin)
    V = spins.shape[0] ** 3
    m = M / V
    return m

def compute_susceptibility(magnetizations, V, T):
    M_abs = np.array(magnetizations) * V  # Total magnetization
    M_abs_mean = np.mean(M_abs)
    M2_mean = np.mean(M_abs ** 2)
    chi = (M2_mean - M_abs_mean ** 2) / (V * T)

    N = len(M_abs)
    numerator = M_abs ** 2 - M_abs_mean ** 2
    sigma_numerator = np.std(numerator, ddof=1)
    error_chi = sigma_numerator / (np.sqrt(N) * V * T)

    return chi, error_chi

def compute_binder_cumulant(magnetizations, V):
    M_abs = np.array(magnetizations) * V
    M2 = M_abs ** 2
    M4 = M_abs ** 4
    U = 1 - np.mean(M4) / (3 * np.mean(M2) ** 2)

    N = len(M_abs)
    sigma_M2 = np.std(M2, ddof=1) / np.sqrt(N)
    sigma_M4 = np.std(M4, ddof=1) / np.sqrt(N)
    dU_dM4 = -1 / (3 * np.mean(M2) ** 2)
    dU_dM2 = (2 * np.mean(M4)) / (3 * np.mean(M2) ** 3)

    # Standard error
    error_U = np.sqrt((dU_dM4 * sigma_M4) ** 2 + (dU_dM2 * sigma_M2) ** 2)
    return U, error_U

def compute_energy(spins, J):
    L = spins.shape[0]
    E = 0.0
    for shift in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        shifted_spins = np.roll(spins, shift=shift, axis=(0, 1, 2))
        dot_product = 2 * np.sum(spins * shifted_spins, axis=-1)
        E -= J * np.sum(dot_product)
    return E  # Each bond is counted twice

def compute_specific_heat(energies, V, T):
    E = np.array(energies)
    N = len(E)
    E_mean = np.mean(E)
    E2_mean = np.mean(E ** 2)
    variance_E = E2_mean - E_mean ** 2
    C = variance_E / (3 * T ** 2)

    # Number of blocks
    M = 10
    m = N // M  # Ensure m is an integer

    # Split energies into M blocks
    variances = []
    for k in range(M):
        E_block = E[k * m:(k + 1) * m]
        E_mean_block = np.mean(E_block)
        E2_mean_block = np.mean(E_block ** 2)
        variance_block = E2_mean_block - E_mean_block ** 2
        variances.append(variance_block)

    variances = np.array(variances)
    variance_mean = np.mean(variances)
    sigma_variance = np.sqrt(np.sum((variances - variance_mean) ** 2) / (M - 1)) / np.sqrt(M)

    # Error in specific heat
    sigma_C = sigma_variance / (V * T ** 2)

    return C, sigma_C

def compute_structure_factor(spins, n_max=5):
    L = spins.shape[0]
    N = L ** 3
    S_q = []
    q_values = []
    spin_complex = spins[..., 0] + 1j * spins[..., 1]

    x = np.arange(L)[:, None, None]
    y = np.arange(L)[None, :, None]
    z = np.arange(L)[None, None, :]

    for n in range(1, n_max + 1):
        q = 2 * np.pi * n / L
        for q_vector in [
            np.array([q, 0, 0]),
            np.array([0, q, 0]),
            np.array([0, 0, q])
        ]:
            r_dot_q = q_vector[0] * x + q_vector[1] * y + q_vector[2] * z
            phase = np.exp(-1j * r_dot_q)
            S_q_value = np.abs(np.sum(spin_complex * phase)) ** 2 / N
            S_q.append(S_q_value)
            q_values.append(np.linalg.norm(q_vector))
    return np.array(S_q), np.array(q_values)

def compute_S0(spins):
    # Convert spins to complex numbers
    spin_complex = spins[..., 0] + 1j * spins[..., 1]  # Shape: (L, L, L)
    N = spins.shape[0] ** 3
    S0 = np.abs(np.sum(spin_complex)) ** 2 / N
    return S0

def estimate_correlation_length(S0, S_q, L):
    q = 2 * np.pi / L  # Smallest non-zero momentum

    # Ensure S_q is not zero to avoid division by zero
    if S_q == 0 or S0 <= S_q:
        return np.inf

    xi = 1 / q * np.sqrt(np.abs(S0 / S_q - 1))

    return xi

def compute_correlation_length_from_cluster_sizes(cluster_sizes, V, num_bins=100):
    # Convert cluster sizes to actual sizes (since they are normalized by V)
    actual_cluster_sizes = np.array(cluster_sizes) * V  # Now cluster sizes are integers

    # Remove zero or negligible cluster sizes (if any)
    actual_cluster_sizes = actual_cluster_sizes[actual_cluster_sizes > 0.5]

    # Build histogram: n_s is the number of clusters of size s
    counts, bin_edges = np.histogram(actual_cluster_sizes, bins=num_bins)

    # Compute bin centers (representative s values)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Avoid zeros in bin centers (to prevent division by zero)
    valid_indices = bin_centers > 0
    bin_centers = bin_centers[valid_indices]
    counts = counts[valid_indices]

    # Compute s^2 * n_s and s^{8/3} * n_s
    s2_ns = bin_centers ** 2 * counts
    s8_3_ns = bin_centers ** (8 / 3) * counts

    # Compute numerator and denominator
    numerator = np.sum(s8_3_ns)
    denominator = np.sum(s2_ns)

    # Check for zero denominator
    if denominator == 0:
        raise ValueError("Denominator in correlation length calculation is zero.")

    # Estimate xi^2 (proportional to numerator/denominator)
    xi_squared = numerator / denominator

    # Compute xi
    xi = np.sqrt(xi_squared)

    return xi

def compute_spin_correlation(spins, max_r):
    """
    Compute spin-spin correlation function G(r).
    """
    L = spins.shape[0]
    G_r = np.zeros(max_r)
    N = L**3

    for r in range(1, max_r + 1):
        corr = 0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    # Periodic boundary conditions
                    ni = (i + r) % L
                    nj = (j + r) % L
                    nk = (k + r) % L
                    corr += np.dot(spins[i, j, k], spins[ni, nj, nk])
        G_r[r - 1] = corr / N

    return G_r

def compute_structure_factor(spins, q_vals):
    """
    Compute structure factor S(q) using Fourier transform.
    """
    L = spins.shape[0]
    N = L**3
    S_q = []

    for q in q_vals:
        # Fourier transform
        spin_sum = np.sum(spins * np.exp(-1j * q * np.arange(L)), axis=(0, 1, 2))
        S_q.append(np.abs(spin_sum)**2 / N)

    return np.array(S_q)

def fit_correlation_length(G_r):
    """
    Fit G(r) to extract correlation length ξ.
    """

    def exponential_decay(r, xi, A):
        return A * np.exp(-r / xi)

    r_vals = np.arange(1, len(G_r) + 1)
    popt, _ = curve_fit(exponential_decay, r_vals, G_r, p0=[1.0, 1.0])
    return popt[0]  # Return ξ

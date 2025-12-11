import numpy as np
from src.xy3d_wolff.simulation import XYStudy
from src.xy3d_wolff.plotter import XYPlotter
from src.xy3d_wolff import core

def main():
    # -------- basic simulation (no estimator) --------
    J = 1.0
    n_steps = 5000
    n_equil = 1000
    L_list = [8,10]
    T_list = np.linspace(1.5, 3.0, 5)
    print("Temperatures:", T_list)

    improved_study = XYStudy(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )

    improved_results = improved_study.run_simulation_all()

    # Fit Critical Temperature From Binder Cumulant
    T_crossing, L_pairs = core.fit_binder_crossings(improved_results, L_list, T_list)
    L_input = []
    Tc = np.mean(T_crossing)

    for i in range(len(L_pairs)):
        L_input.append(L_pairs[i][0])
    L_input = np.array(L_input)

    print("L_input: ", L_input)
    Tc, nu, nu_error, c = core.scaling_fit(L_input)
    print(f"Fitted ν: {nu:.4f}")
    print(f"Fitted Tc: {Tc:.4f}")
    print("c: ", c)

    # Fit susceptibility to get γ
    chi_fit_results = core.fit_susceptibility0(improved_results, L_list, T_list, Tc=Tc, plot=True)

    # Fit magnetization per L
    m_fit_results = core.fit_magnetization_per_L(improved_results, L_list, T_list, Tc=Tc, plot=True)
    beta_list = []
    beta_error_list = []
    for L in L_list:
        beta, beta_error = m_fit_results[L]['beta']
        beta_list.append(beta)
        beta_error_list.append(beta_error)
    beta_list = np.array(beta_list)
    beta_error_list = np.array(beta_error_list)
    print("beta_error_list: ", beta_error_list)

    # Fit specific heat per L to get α
    C_fit_results = core.fit_specific_heat_per_L(improved_results, L_list, T_list, Tc=Tc, plot=True)
    alpha_list = []
    alpha_error_list = []
    for L in L_list:
        alpha, alpha_error = C_fit_results[L]['alpha']
        alpha_list.append(alpha)
        alpha_error_list.append(alpha_error)
    alpha_list = np.array(alpha_list)
    alpha_error_list = np.array(alpha_error_list)
    print("alpha_list: ", alpha_list)
    print("alpha_error_list: ", alpha_error_list)

    # Fit susceptibility from nu
    gamma, gamma_over_nu, fit_details = core.fit_susceptibility(improved_results, T_list, L_list, nu)
    # Assume you have the following variables from your fittings:
    # From susceptibility fitting:
    gamma_over_nu = fit_details['gamma_over_nu']
    gamma_over_nu_error = fit_details['gamma_over_nu_error']

    # From prior nu fitting:
    nu_error = nu_error  # Standard deviation of nu from your prior fitting

    # Calculate the standard deviation of gamma using error propagation
    gamma_error = gamma * np.sqrt(
        (gamma_over_nu_error / gamma_over_nu) ** 2 +
        (nu_error / nu) ** 2
    )

    # Print the results
    print(f"Estimated γ = {gamma:.3f} ± {nu_error[0]:.3f}")
    print("nu_error: ", nu_error)

    # Fit specific heat
    alpha, alpha_over_nu, fit_details = core.fit_specific_heat(improved_results, T_list, L_list, nu)
    # Assume you have the following variables from your fittings:
    # From susceptibility fitting:
    alpha_over_nu = fit_details['alpha_over_nu']
    alpha_over_nu_error = fit_details['alpha_over_nu_error']

    # From prior nu fitting:
    nu_error = nu_error  # Standard deviation of nu from your prior fitting

    # Calculate the standard deviation of gamma using error propagation
    alpha_error = alpha * np.sqrt(
        (alpha_over_nu_error / alpha_over_nu) ** 2 +
        (nu_error / nu) ** 2
    )

    # Print the results
    print(f"Estimated alpha = {alpha:.3f} ± {alpha_error[2]:.3f}")


    # Fit magnetization
    beta, beta_over_nu, fit_details = core.fit_magnetization(improved_results, T_list, L_list, nu)
    # Assume you have the following variables from your fittings:
    # From susceptibility fitting:
    beta_over_nu = fit_details['beta_over_nu']
    beta_over_nu_error = fit_details['beta_over_nu_error']

    # From prior nu fitting:
    nu_error = nu_error  # Standard deviation of nu from your prior fitting

    # Calculate the standard deviation of gamma using error propagation
    beta_error = beta * np.sqrt(
        (beta_over_nu_error / beta_over_nu) ** 2 +
        (nu_error / nu) ** 2
    )

    # Print the results
    print(f"Estimated beta = {beta:.3f} ± {beta_error[0]:.3f}")

    # Fit nu from binder cumulant
    Tc = np.mean(T_crossing)
    nu_est, nu_err, fit_details = core.fit_nu_from_binder_cumulant(improved_results, L_list, T_list, Tc)
    print("nu_est: ", nu_est)
    print("nu_error: ", nu_err)
    print("T_list: ", T_list)
    print("Tc: ", Tc)

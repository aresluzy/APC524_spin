from __future__ import annotations
from typing import Dict, Any, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


class XYPlotter:
    """
    Plotting utilities for XY Monte Carlo simulations.
    """

    # ------------------------------------------------------------------
    # 1. Plot the observables from simulation results
    # ------------------------------------------------------------------
    @staticmethod
    def plot_observables(simulation_results, L_list, T_list, T_c = 2.2):
        """
        Plot magnetization, energy, cluster size, susceptibility, specific heat,
        and Binder cumulant vs temperature.

        Parameters
        ----------
        simulation_results : dict
            Nested results: simulation_results[L][T] with keys
            'magnetizations', 'energies', 'cluster_sizes',
            'susceptibility', 'specific_heat', 'binder_cumulant'.
        L_list : sequence of int
            System sizes.
        T_list : sequence of float
            Temperatures.
        T_c : float, optional
            Critical temperature used for the red vertical line.
        """
        # Plot Magnetization
        plt.figure()
        for L in L_list:
            temperatures = []
            avg_magnetizations = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                avg_magnetizations.append(np.mean(results["magnetizations"]))

            plt.plot(temperatures, avg_magnetizations, "o-", label=f"L={L}")
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Magnetization vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Magnetization")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Energy
        plt.figure()
        for L in L_list:
            temperatures = []
            avg_energies = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                avg_energies.append(np.mean(results["energies"]))

            plt.plot(temperatures, avg_energies, "o-", label=f"L={L}")
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Energy vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Cluster Size
        plt.figure()
        for L in L_list:
            temperatures = []
            avg_cluster_sizes = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                avg_cluster_sizes.append(np.mean(results["cluster_sizes"]))

            plt.plot(temperatures, avg_cluster_sizes, "o-", label=f"L={L}")
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Cluster Size vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Cluster Size")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Susceptibility
        plt.figure()
        for L in L_list:
            temperatures = []
            susceptibilities = []
            susceptibility_errors = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                susceptibilities.append(results["susceptibility"][0])
                susceptibility_errors.append(results["susceptibility"][1])

            plt.errorbar(
                temperatures,
                susceptibilities,
                yerr=susceptibility_errors,
                fmt="o-",
                label=f"L={L}",
            )
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Susceptibility vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Susceptibility")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Specific Heat
        plt.figure()
        for L in L_list:
            temperatures = []
            specific_heats = []
            specific_heat_errors = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                specific_heats.append(results["specific_heat"][0])
                specific_heat_errors.append(results["specific_heat"][1])

            plt.errorbar(
                temperatures,
                specific_heats,
                yerr=specific_heat_errors,
                fmt="o-",
                label=f"L={L}",
            )
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Specific Heat vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Specific Heat")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Binder Cumulant
        plt.figure()
        for L in L_list:
            temperatures = []
            binder_cumulants = []
            binder_cumulant_errors = []

            for T in T_list:
                results = simulation_results[L][T]
                temperatures.append(T)
                binder_cumulants.append(results["binder_cumulant"][0])
                binder_cumulant_errors.append(results["binder_cumulant"][1])

            plt.errorbar(
                temperatures,
                binder_cumulants,
                yerr=binder_cumulant_errors,
                fmt="o-",
                label=f"L={L}",
            )
        plt.axvline(x=T_c, color="r", linestyle="--", label=f"T_c = {T_c}")
        plt.title("Binder Cumulant vs Temperature")
        plt.xlabel("Temperature (T)")
        plt.ylabel("Binder Cumulant")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Correlation Length
        # plt.figure()
        # for L in L_list:
        #     temperatures = []
        #     Correlation_length = []
        #     Correlation_length_errors = []

        #     for T in T_list:
        #         results = simulation_results[L][T]
        #         temperatures.append(T)
        #         Correlation_length.append(results['correlation_length'][0])
        #         Correlation_length_errors.append(results['correlation_length'][1])

        #     plt.errorbar(temperatures, Correlation_length, yerr=Correlation_length_errors, fmt='o-', label=f"L={L}")
        # plt.axvline(x=T_c, color='r', linestyle='--', label=f"T_c = {T_c}")
        # plt.title("Correlation Length vs Temperature")
        # plt.xlabel("Temperature (T)")
        # plt.ylabel("Correlation Length")
        # plt.legend()
        # plt.grid()
        # plt.show()

    # ------------------------------------------------------------------
    # 2. Plot 3D spin orientations
    # ------------------------------------------------------------------
    @staticmethod
    def plot_spin_orientations(simulation_results, L, T):
        """
        Plots the 3D spin orientations for a specific lattice size L
        at a specific temperature T.

        Parameters
        ----------
        simulation_results : dict
            {L: {T: {'spin': ndarray of shape (L, L, L, 2)}, ...}, ...}.
        L : int
            Lattice size.
        T : float
            Temperature.
        """
        # Extract spin data
        if L not in simulation_results or T not in simulation_results[L]:
            print(f"No data available for L={L} and T={T}.")
            return

        spins = simulation_results[L][T]['spin']  # Expected shape: (L, L, L, 2)

        # Validate spins shape
        if spins.shape[:3] != (L, L, L) or spins.shape[-1] != 2:
            print(f"Invalid spins data shape: {spins.shape}. Expected (L, L, L, 2).")
            return

        # Generate grid points for lattice
        x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')

        # Compute spin components
        u = spins[..., 0]  # x-component of spin
        v = spins[..., 1]  # y-component of spin
        w = np.zeros_like(u)  # z-component is zero for XY model

        # Create the 3D quiver plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True, color='blue', alpha=0.8)

        # Label axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Spin Orientations (L={L}, T={T})')

        # Set limits
        ax.set_xlim([0, L - 1])
        ax.set_ylim([0, L - 1])
        ax.set_zlim([0, L - 1])

        plt.show()

    # ------------------------------------------------------------------
    # 3. Create spin gif
    # ------------------------------------------------------------------
    @staticmethod
    def create_spin_orientation_gif(simulation_results, L, T_list, interval, output_file="spin_orientations.gif"):
        """
        Creates a GIF showing 3D spin orientations for all temperatures in T_list.

        Parameters:
            simulation_results (dict): Simulation results containing spin data.
                                       Format: {L: {T: {'spins': ndarray of shape (L, L, L, 2)}, ...}, ...}.
            L (int): Lattice size.
            T_list (list): List of temperatures.
            output_file (str): Filename for the output GIF.
            interval (int): Time interval (ms) between frames in the GIF.
        """
        temp_dir = "frames"
        os.makedirs(temp_dir, exist_ok=True)
        frames = []

        for T in T_list:
            # Check if data for L and T exists
            if L not in simulation_results or T not in simulation_results[L]:
                print(f"No data available for L={L} and T={T}. Skipping...")
                continue

            spins = simulation_results[L][T].get('spin', None)
            if spins is None:
                print(f"Spin data not found for L={L} and T={T}. Skipping...")
                continue

            # Validate spins shape
            if spins.shape[:3] != (L, L, L) or spins.shape[-1] != 2:
                print(f"Invalid spins data shape for L={L} and T={T}: {spins.shape}. Expected (L, L, L, 2).")
                continue

            # Generate grid points for lattice
            x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')

            # Compute spin components
            u = spins[..., 0]  # x-component of spin
            v = spins[..., 1]  # y-component of spin
            w = np.zeros_like(u)  # z-component is zero for XY model

            # Create the 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True, color='blue', alpha=0.8)

            # Label axes and title
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title(f'Spin Orientations (L={L}, T={T})')

            # Set limits
            ax.set_xlim([0, L - 1])
            ax.set_ylim([0, L - 1])
            ax.set_zlim([0, L - 1])

            # Save the current frame
            frame_path = os.path.join(temp_dir, f"frame_{T}.png")
            plt.savefig(frame_path)
            frames.append(frame_path)
            plt.close(fig)

        # Create GIF
        images = [imageio.imread(frame) for frame in frames]
        imageio.mimsave(output_file, images, duration=interval)

        # Cleanup temporary files
        for frame in frames:
            os.remove(frame)
        os.rmdir(temp_dir)

        print(f"GIF saved as {output_file}")


class DataCollapsePlotter:
    @staticmethod
    def data_collapse_specific_heat(simulation_results, L_list, T_list, Tc, alpha, nu, plot=True):
        """
        Perform finite-size–scaling data collapse for the specific heat C(T, L).

        This computes the scaled variables

            x = (T - Tc) * L^(1/nu)
            y = C(T,L) / L^(alpha/nu)

        and overlays all system sizes on a single universal curve.

        Parameters
        ----------
        simulation_results : dict
            Nested results dictionary from the simulation:
            ``simulation_results[L][T]['specific_heat'] → (C_mean, C_err)``.
        L_list : sequence of int
            List of system sizes.
        T_list : sequence of float
            Temperatures simulated at each system size.
        Tc : float
            Critical temperature used for rescaling.
        alpha : float
            Critical exponent α for the specific heat divergence.
        nu : float
            Correlation-length exponent ν.
        plot : bool, optional
            If True (default), produce a collapse scatter plot.

        Returns
        -------
        dict
            Dictionary mapping each L to a tuple ``(rescaled_T, rescaled_C)``,
            where rescaled arrays have shapes matching ``T_list``.
        """
        collapsed_data = {}
        step = -1

        for L in L_list:
            # Collect data for this lattice size
            T_all = []
            C_all = []
            step += 1
            for T in T_list:
                results = simulation_results[L][T]
                C_mean, _ = results['specific_heat']
                T_all.append(T)
                C_all.append(C_mean)

            T_all = np.array(T_all)
            C_all = np.array(C_all)
            # Rescale data
            rescaled_T = (T_all - Tc) * L ** (1 / nu)
            rescaled_C = C_all / L ** (alpha / nu)

            collapsed_data[L] = (rescaled_T, rescaled_C)

            # Plot collapsed data
            if plot:
                plt.scatter(rescaled_T, rescaled_C, label=f'L={L}', s=10)

        if plot:
            plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
            plt.xlabel(r'Rescaled Temperature $(T - T_c) L^{1/\nu}$')
            plt.ylabel(r'Rescaled Specific Heat $C(T, L) / L^{\alpha/\nu}$')
            plt.title('Specific Heat Data Collapse')
            plt.legend()
            plt.grid(True)
            plt.show()

        return collapsed_data

    @staticmethod
    def data_collapse_susceptibility(simulation_results, L_list, T_list, Tc, gamma, nu, plot=True):
        """
        Perform finite-size–scaling data collapse for the susceptibility χ(T,L).

        This computes the scaled variables

            x = (T - Tc) * L^(1/nu)
            y = χ(T,L) / L^(gamma/nu)

        and overlays data from all lattice sizes.

        Parameters
        ----------
        simulation_results : dict
            Simulation output dictionary with entries:
            ``simulation_results[L][T]['susceptibility'] → (chi_mean, chi_err)``.
        L_list : sequence of int
            List of system sizes.
        T_list : sequence of float
            Temperatures simulated for each L.
        Tc : float
            Critical temperature used for rescaling.
        gamma : float
            Susceptibility critical exponent γ.
        nu : float
            Correlation-length exponent ν.
        plot : bool, optional
            If True (default), generate a collapse plot.

        Returns
        -------
        dict
            Mapping ``L → (rescaled_T, rescaled_chi)``.
        """

        collapsed_data = {}
        step = -1
        for L in L_list:
            # Collect susceptibility data for this lattice size
            T_all = []
            chi_all = []
            step += 1
            for T in T_list:
                results = simulation_results[L][T]
                chi_mean, _ = results['susceptibility']  # Extract susceptibility
                T_all.append(T)
                chi_all.append(chi_mean)

            T_all = np.array(T_all)
            chi_all = np.array(chi_all)

            # Rescale data
            rescaled_T = (T_all - Tc) * L ** (1 / nu)
            rescaled_chi = chi_all / L ** (gamma / nu)

            collapsed_data[L] = (rescaled_T, rescaled_chi)

            # Plot collapsed data
            if plot:
                plt.scatter(rescaled_T, rescaled_chi, label=f'L={L}', s=10)

        if plot:
            plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
            plt.xlabel(r'Rescaled Temperature $(T - T_c) L^{1/\nu}$')
            plt.ylabel(r'Rescaled Susceptibility $\chi(T, L) / L^{\gamma/\nu}$')
            plt.title('Magnetic Susceptibility Data Collapse')
            plt.legend()
            plt.grid(True)
            plt.show()

        return collapsed_data

    @staticmethod
    def data_collapse_magnetization(simulation_results, L_list, T_list, Tc, beta, nu, plot=True):
        """
        Perform finite-size–scaling data collapse for magnetization M(T,L).

        Uses the standard FSS rescaling:

            x = (Tc - T) * L^(1/nu)
            y = M(T,L) * L^(beta/nu)

        valid for T < Tc.

        Parameters
        ----------
        simulation_results : dict
            Simulation dictionary such that:
            ``simulation_results[L][T]['magnetizations']`` contains an array
            of magnetization measurements.
        L_list : sequence of int
            System sizes included in the collapse.
        T_list : sequence of float
            Temperatures at which observables were measured.
        Tc : float
            Critical temperature.
        beta : float
            Magnetization critical exponent β.
        nu : float
            Correlation-length exponent ν.
        plot : bool, optional
            If True (default), produce a data-collapse plot.

        Returns
        -------
        dict
            Dictionary mapping each L to
            ``(rescaled_T, rescaled_M)``.
        """
        collapsed_data = {}
        step = 0
        for L in L_list:
            # Collect magnetization data for this lattice size
            T_all = []
            M_all = []
            for T in T_list:
                results = simulation_results[L][T]
                M_mean = np.mean(np.abs(results['magnetizations']))  # Use absolute magnetization
                T_all.append(T)
                M_all.append(M_mean)

            T_all = np.array(T_all)
            M_all = np.array(M_all)

            # Rescale data
            rescaled_T = (Tc - T_all) * L ** (1 / nu)
            rescaled_M = M_all * L ** (beta / nu)

            collapsed_data[L] = (rescaled_T, rescaled_M)
            step += 1
            # Plot collapsed data
            if plot:
                plt.scatter(rescaled_T, rescaled_M, label=f'L={L}', s=10)

        if plot:
            plt.axvline(0, color='red', linestyle='--', label=f'T_c={Tc}')
            plt.xlabel(r'Rescaled Temperature $(T_c - T) L^{1/\nu}$')
            plt.ylabel(r'Rescaled Magnetization $M(T, L) \cdot L^{\beta/\nu}$')
            plt.title('Magnetization Data Collapse')
            plt.legend()
            plt.grid(True)
            plt.show()

        return collapsed_data
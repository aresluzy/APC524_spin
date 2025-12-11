import numpy as np
from src.xy3d_wolff.simulation import XYStudy
from src.xy3d_wolff.plotter import XYPlotter


def main():
    # -------- basic simulation (no estimator) --------
    J = 1.0
    n_steps = 5000
    n_equil = 1000
    L_list = [8,10,12,14,16,18,20]
    T_list = np.linspace(1.5, 3.0, 20)
    print("Temperatures:", T_list)

    basic_study = XYStudy(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )

    basic_results = basic_study.run_simulation_all()
    XYPlotter.plot_observables(basic_results, L_list, T_list)
    XYPlotter.plot_spin_orientations(basic_results, 8, 1.5)
    XYPlotter.create_spin_orientation_gif(basic_results, 8, T_list, 500, output_file="spin_orientations.gif")


if __name__ == "__main__":
    main()

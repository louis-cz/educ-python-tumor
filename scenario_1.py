from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":

    sim_params = {
        'temps_cc': 24,
        'taille': 11,
        'n_jours': 1000,
        'p_apoptose': 0,
        'mu': 10
    }

    img_intervals = {
        10: [4, 8, 12],
        15: [16, 33, 50],
        20: [260, 520, 781]
    }

    pmax_val = [10, 15, 20] # valeurs de pmax à tester
    pinit = [11, 16, 21] # valeurs de pinit correspondantes (pmax + 1 car on veut des RTC au départ)
    n_sim_per_pmax = 3 # nombre de simulations par pmax

    results = {
        10: [],
        15: [],
        20: []
    }

    for pmax, pinit_val in zip(pmax_val, pinit):
        sim_params['pmax'] = pmax
        sim_params['pinit'] = pinit_val

        for sim_idx in range(n_sim_per_pmax):
            print(f"\n--- Simulation pour pmax = {pmax}, Simulation {sim_idx + 1}/{n_sim_per_pmax} ---")
            if sim_idx == 0: # sauvegarder les images uniquement pour la première simulation de chaque pmax
                grilles = simulation(sim_params, save_img=True, img_itrvl=img_intervals[pmax], img_dir=f"img/S1_pmax_{pmax}")
            else:
                grilles = simulation(sim_params, save_img=False)
            results[pmax].append(grilles)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    pop_dynamics = pop_vs_time(results, pmax_val, colors=colors, pmax=pmax_val, log_scale=True, pop="total", legend_prefix="pmax")
    pop_dynamics.savefig("plots/S1_impact_pmax_population_tumorale_totale.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, pmax_val, colors=colors, pmax=pmax_val, log_scale=True, pop="rtc", legend_prefix="pmax")
    pop_dynamics_RTC.savefig("plots/S1_impact_pmax_population_tumorale_RTC.png", dpi=150)
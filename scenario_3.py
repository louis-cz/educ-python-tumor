from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":
    
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 220,
        'p_apoptose': 0,
        'p_stc' : 0.05,
        'mu': 10
    }

    img_intervals = {
        5: [69, 138, 208],
        10: [69, 138, 208],
        15: [69, 138, 208],
        20: [69, 138, 208]
    }

    pmax_val = [5, 10, 15, 20] # valeurs de pmax à tester
    pinit = [8, 13, 18, 23] # valeurs de pinit correspondantes (pmax + 3 car on veut une true STC au départ)
    n_sim_per_pmax = 3 # nombre de simulations par pmax

    results = {
        5: [],
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
                grilles = simulation(sim_params, save_img=True, img_itrvl=img_intervals[pmax], img_dir=f"img/S3_pmax_{pmax}")
            else:
                grilles = simulation(sim_params, save_img=False)
            results[pmax].append(grilles)

    pop_dynamics_STC = pop_vs_time(results, pmax_val, pmax_val, colors=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"], pop="stc")
    pop_dynamics_STC.savefig("plots/S3_impact_pmax_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, pmax_val, pmax_val, colors=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"], pop="rtc")
    pop_dynamics_RTC.savefig("plots/S3_impact_pmax_population_tumorale_RTC.png", dpi=150)
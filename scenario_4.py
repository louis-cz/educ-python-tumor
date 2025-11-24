from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":
    
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 420,
        'p_stc' : 0.01,
        'mu': 10,
        'pmax': 10,
        'pinit': 13,
    }
    
    img_intervals = [138, 277, 416]
    p_apoptose_values = [0, 0.01, 0.1, 0.3] # valeurs de papt à tester
    n_sim_per_papt = 3 # nombre de simulations par valeur de papt

    results = {
        0: [],
        0.01: [],
        0.1: [],
        0.3: []
    }

    for p_apoptose in p_apoptose_values:
        sim_params['p_apoptose'] = p_apoptose

        for sim_idx in range(n_sim_per_papt):
            print(f"\n--- Simulation pour p_apoptose = {p_apoptose}, Simulation {sim_idx + 1}/{n_sim_per_papt} ---")
            if sim_idx == 0: # sauvegarder les images uniquement pour la première simulation de chaque pmax
                grilles = simulation(sim_params, save_img=True, img_itrvl=img_intervals, img_dir=f"img/S4_papt_{p_apoptose}")
            else:
                grilles = simulation(sim_params, save_img=False)
            results[p_apoptose].append(grilles)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c",  "#d62728"]

    pop__dynamics = pop_vs_time(results, p_apoptose_values, colors=colors, pmax=10, pop="total", legend_prefix="papt")
    pop__dynamics.savefig("plots/S4_impact_papt_population_tumorale_totale.png", dpi=150)

    pop_dynamics_STC = pop_vs_time(results, p_apoptose_values, colors=colors, pmax=10, pop="stc", legend_prefix="papt")
    pop_dynamics_STC.savefig("plots/S4_impact_papt_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, p_apoptose_values, colors=colors, pmax=10, pop="rtc", legend_prefix="papt")
    pop_dynamics_RTC.savefig("plots/S4_impact_papt_population_tumorale_RTC.png", dpi=150)
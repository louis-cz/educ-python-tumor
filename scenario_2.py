from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":
    
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 200,
        'p_apoptose': 0,
        'mu': 10
    }

    img_intervals = {
        10: [41, 83, 125],
        15: [41, 83, 125],
        20: [41, 83, 125]
    }

    pmax_val = [10, 15, 20] # valeurs de pmax à tester
    pinit = [12, 17, 22] # valeurs de pinit correspondantes (pmax + 2 car on veut une STC clonogenic au départ)
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
                grilles = simulation(sim_params, save_img=True, img_itrvl=img_intervals[pmax], img_dir=f"img/S2_pmax_{pmax}")
            else:
                grilles = simulation(sim_params, save_img=False)
            results[pmax].append(grilles)

    pop_dynamics = pop_vs_time(results, pmax_val, pmax_val, colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
    pop_dynamics.savefig("plots/S2_impact_pmax_population_tumorale.png", dpi=150)

from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 50,
        'p_apoptose': 0,
        'mu': 10
    }

    # intervalles de sauvegarde des images pour chaque pmax
    img_intervals = {
        10: [41, 83, 125],
        15: [41, 83, 125],
        20: [41, 83, 125]
    }

    pmax = [10, 15, 20] # valeurs de pmax à tester
    pinit = [12, 17, 22] # valeurs de pinit correspondantes (pmax + 2 car on veut une STC clonogenic au départ)
    n_sim_per_pmax = 3 # nombre de simulations par pmax
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] # couleurs pour les plots

    results = {f"pmax={pmax_val}": [] for pmax_val in pmax} # liste pour chaque condition avec clef = "pmax=valeur"

    for pmax_val, pinit_val in zip(pmax, pinit):
        sim_params.update({'pmax': pmax_val, 'pinit': pinit_val}) # mise à jour des paramètres de simulation

        for sim_idx in range(n_sim_per_pmax):
            print(f"\n--- Simulation pour pmax = {pmax_val}, Simulation {sim_idx + 1}/{n_sim_per_pmax} ---")
            save_img = (sim_idx == 0) # sauvegarder les images seulement pour la première simulation de chaque pmax
            img_args = { 
                "save_img": save_img,
                "img_itrvl": img_intervals[pmax_val] if save_img else None,
                "img_dir": f"img/Sdummy_pmax_{pmax_val}" if save_img else None,
                "save_json": save_img
            }
            cell_counts = simulation(sim_params, **img_args)
            results[f"pmax={pmax_val}"].append(cell_counts)

    pop_dynamics = pop_vs_time(results, colors=colors, pop="total", log_scale=False)
    pop_dynamics.savefig("plots/Sdummy_impact_pmax_population_tumorale_totale.png", dpi=150)

    pop_dynamics_STC = pop_vs_time(results, colors=colors, pop="stc", log_scale=False)
    pop_dynamics_STC.savefig("plots/Sdummy_impact_pmax_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, colors=colors, pop="rtc", log_scale=False)
    pop_dynamics_RTC.savefig("plots/Sdummy_impact_pmax_population_tumorale_RTC.png", dpi=150)
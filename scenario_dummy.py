from simulation_tumor import simulation, pop_vs_time
import numpy as np

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 5,
        'n_jours': 100,
        'p_stc' : 0.1,
        'mu': 10,
        'pmax': 10,
        'pinit': 13,
    }
    
    img_intervals = [138, 277, 416] # intervalles de sauvegarde des images
    p_apoptose_values = [0, 0.01, 0.1, 0.3] # valeurs de papt à tester
    n_sim_per_papt = 3 # nombre de simulations par valeur de papt
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c",  "#d62728"] # couleurs pour les plots

    results = {p_apoptose: [] for p_apoptose in p_apoptose_values} # liste pour chaque condition avec clef = 

    for p_apoptose in p_apoptose_values:
        sim_params['p_apoptose'] = p_apoptose # mise à jour des paramètres de simulation

        for sim_idx in range(n_sim_per_papt):
            print(f"\n--- Simulation pour p_apoptose = {p_apoptose}, Simulation {sim_idx + 1}/{n_sim_per_papt} ---")
            save_img = (sim_idx == 0) # sauvegarder les images seulement pour la première simulation de chaque p_apoptose
            img_args = { 
                "save_img": save_img,
                "img_itrvl": img_intervals if save_img else None,
                "img_dir": f"img/Sdummy_papt_{p_apoptose}" if save_img else None,
                "save_json": save_img
            }
            cell_counts = simulation(sim_params, **img_args)
            results[p_apoptose].append(cell_counts)

    pop_dynamics = pop_vs_time(results, colors=colors, pop="total", log_scale=False)
    pop_dynamics.savefig("plots/Sdummy_impact_papt_population_tumorale_totale.png", dpi=150)

    pop_dynamics_STC = pop_vs_time(results, colors=colors, pop="stc", log_scale=False)
    pop_dynamics_STC.savefig("plots/Sdummy_impact_papt_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, colors=colors, pop="rtc", log_scale=False)
    pop_dynamics_RTC.savefig("plots/Sdummy_impact_papt_population_tumorale_RTC.png", dpi=150)
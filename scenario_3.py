from simulation_tumor import simulation, pop_vs_time
import pandas as pd
import numpy as np


if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 220,
        'p_apoptose': 0,
        'p_stc' : 0.05,
        'mu': 10
    }

    # intervalles de sauvegarde des images pour chaque pmax
    img_intervals = {
        5: [69, 138, 208],
        10: [69, 138, 208],
        15: [69, 138, 208],
        20: [69, 138, 208]
    }

    pmax = [5, 10, 15, 20] # valeurs de pmax à tester
    pinit = [8, 13, 18, 23] # valeurs de pinit correspondantes (pmax + 3 car on veut une true STC au départ)
    n_sim_per_pmax = 5 # nombre de simulations par pmax
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"] # couleurs pour les plots

    results = {pmax_val: [] for pmax_val in pmax} # liste pour chaque condition avec clef = pmax
    final_counts = [] # pour stocker les moyennes et écarts-types

    for pmax_val, pinit_val in zip(pmax, pinit):
        sim_params.update({'pmax': pmax_val, 'pinit': pinit_val}) # mise à jour des paramètres de simulation

        for sim_idx in range(n_sim_per_pmax):
            print(f"\n--- Simulation pour pmax = {pmax_val}, Simulation {sim_idx + 1}/{n_sim_per_pmax} ---")
            save_img = (sim_idx == 0) # sauvegarder les images seulement pour la première simulation de chaque pmax
            img_args = { 
                "save_img": save_img,
                "img_itrvl": img_intervals[pmax_val] if save_img else None,
                "img_dir": f"img/S3_pmax_{pmax_val}" if save_img else None,
                "save_json": save_img
            }
            cell_counts = simulation(sim_params, **img_args)
            results[pmax_val].append(cell_counts)
        
        final_rtc_counts = [sim['rtc'][-1] for sim in results[pmax_val]]
        final_stc_counts = [sim['stc'][-1] for sim in results[pmax_val]]
        rtc_mean = np.mean(final_rtc_counts)
        rtc_std = np.std(final_rtc_counts)
        stc_mean = np.mean(final_stc_counts)
        stc_std = np.std(final_stc_counts)
        final_counts.append({
            'pmax': pmax_val,
            'rtc_mean': rtc_mean,
            'rtc_std': rtc_std,
            'stc_mean': stc_mean,
            'stc_std': stc_std
        })

    pop_dynamics = pop_vs_time(results, colors=colors, pop="total", prefix="pmax")
    pop_dynamics.savefig("plots/S3_impact_pmax_population_tumorale_totale.png", dpi=150)

    pop_dynamics_STC = pop_vs_time(results, colors=colors, pop="stc", prefix="pmax")
    pop_dynamics_STC.savefig("plots/S3_impact_pmax_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, colors=colors, pop="rtc", prefix="pmax")
    pop_dynamics_RTC.savefig("plots/S3_impact_pmax_population_tumorale_RTC.png", dpi=150)

    print("\n--- Résultats finaux des populations cellulaires ---")
    table_rows = []
    for row in final_counts:
        print(f"pmax = {row['pmax']}: RTC mean = {row['rtc_mean']:.2f} ± {row['rtc_std']:.2f}, STC mean = {row['stc_mean']:.2f} ± {row['stc_std']:.2f}")
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    df.to_csv("data/scenario_3_results.csv", index=False)
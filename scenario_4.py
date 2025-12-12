from simulation_tumor import simulation, pop_vs_time
import pandas as pd
import numpy as np


if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 300,
        'p_stc' : 0.05,
        'mu': 10,
        'pmax': 10,
        'pinit': 13,
    }
    
    img_intervals = [138, 277, 416] # intervalles de sauvegarde des images
    p_apoptose_values = [0, 0.01, 0.1, 0.3] # valeurs de papt à tester
    n_sim_per_papt = 3 # nombre de simulations par valeur de papt
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c",  "#d62728"] # couleurs pour les plots

    results = {p_apoptose: [] for p_apoptose in p_apoptose_values} # liste pour chaque condition avec clef = p_apoptose
    final_counts = [] # pour stocker les moyennes et écarts-types

    for p_apoptose in p_apoptose_values:
        sim_params['p_apoptose'] = p_apoptose # mise à jour des paramètres de simulation

        for sim_idx in range(n_sim_per_papt):
            print(f"\n--- Simulation pour p_apoptose = {p_apoptose}, Simulation {sim_idx + 1}/{n_sim_per_papt} ---")
            save_img = (sim_idx == 0) # sauvegarder les images seulement pour la première simulation de chaque p_apoptose
            img_args = { 
                "save_img": save_img,
                "img_itrvl": img_intervals if save_img else None,
                "img_dir": f"img/S4_papt_{p_apoptose}" if save_img else None,
                "save_json": save_img
            }
            cell_counts = simulation(sim_params, **img_args)
            results[p_apoptose].append(cell_counts)

        final_rtc_counts = [sim['rtc'][-1] for sim in results[p_apoptose]]
        final_stc_counts = [sim['stc'][-1] for sim in results[p_apoptose]]
        rtc_mean = np.mean(final_rtc_counts)
        rtc_std = np.std(final_rtc_counts)
        stc_mean = np.mean(final_stc_counts)
        stc_std = np.std(final_stc_counts)
        final_counts.append({
            'p_apoptose': p_apoptose,
            'rtc_mean': rtc_mean,
            'rtc_std': rtc_std,
            'stc_mean': stc_mean,
            'stc_std': stc_std
        })

    pop_dynamics = pop_vs_time(results, colors=colors, pop="total", prefix="p_apt")
    pop_dynamics.savefig("plots/S4_impact_papt_population_tumorale_totale.png", dpi=150)

    pop_dynamics_STC = pop_vs_time(results, colors=colors, pop="stc", prefix="p_apt")
    pop_dynamics_STC.savefig("plots/S4_impact_papt_population_tumorale_STC.png", dpi=150)
    
    pop_dynamics_RTC = pop_vs_time(results, colors=colors, pop="rtc", prefix="p_apt")
    pop_dynamics_RTC.savefig("plots/S4_impact_papt_population_tumorale_RTC.png", dpi=150)

    print("\n--- Résultats finaux des populations cellulaires ---")
    table_rows = []
    for row in final_counts:
        print(f"p_apoptose = {row['p_apoptose']}: RTC mean = {row['rtc_mean']:.2f} ± {row['rtc_std']:.2f}, STC mean = {row['stc_mean']:.2f} ± {row['stc_std']:.2f}")
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    df.to_csv("data/scenario_4_results.csv", index=False)

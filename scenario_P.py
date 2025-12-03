from simulation_tumor_immune import simulation, pop_vs_time
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        # généraux
        'n_jours': 200,
        'taille': 100,
        # tumoraux
        "pmax_t": 10,
        "pinit_t": 12,
        "p_apoptose_t": 0.01,
        "p_proliferation_t": 0.1,
        "p_stc_t": 0.01,
        "mu_t": 5,
        # immunitaires
        "pmax_i": -5,
        "pinit_i": -5,
        "p_apoptose_i": 0.01,
        "p_proliferation_i": 0.025,
        "p_phagocytose_i": 0.5, # par jour
        "mu_i": 10,
    }

    n_cells_i = [5, 10, 20]
    n_sim_per_pmax = 3 # nombre de simulations par pmax
    results = {n_cells_i_val: [] for n_cells_i_val in n_cells_i}
    final_counts = []
    colors_i = ["#2ca02c", "#17becf", "#bcbd22"]
    colors_t = ["#d62728", "#ff7f0e", "#1f77b4"]
    img_intervals = {
        5: [10, 20, 30, 50, 70, 100, 150, 200],
        10: [10, 20, 30, 50, 70, 100, 150, 200],
        20: [10, 20, 30, 50, 70, 100, 150, 200],
    }
    
    for n_cells_i_val in n_cells_i:
        sim_params['n_cells_i'] = n_cells_i_val

        for sim_idx in range(n_sim_per_pmax):
            print(f"\n--- Simulation pour n_cells_i = {n_cells_i_val}, Simulation {sim_idx + 1}/{n_sim_per_pmax} ---")
            save_img = (sim_idx == 0) # sauvegarder les images seulement pour la première simulation de chaque pmax
            img_args = { 
                "save_img": save_img,
                "img_itrvl": img_intervals[n_cells_i_val] if save_img else None,
                "img_dir": f"img/SP_n_cells_i_{n_cells_i_val}" if save_img else None
            }
            cell_counts = simulation(sim_params, **img_args)
            results[n_cells_i_val].append(cell_counts)
        
        final_rtc_counts = [sim['rtc'][-1] for sim in results[n_cells_i_val]]
        final_stc_counts = [sim['stc'][-1] for sim in results[n_cells_i_val]]
        final_immune_counts = [sim['immune'][-1] for sim in results[n_cells_i_val]]
        rtc_mean = np.mean(final_rtc_counts)
        rtc_std = np.std(final_rtc_counts)
        stc_mean = np.mean(final_stc_counts)
        stc_std = np.std(final_stc_counts)
        immune_mean = np.mean(final_immune_counts)
        immune_std = np.std(final_immune_counts)
        final_counts.append({
            'n_cells_i': n_cells_i_val,
            'rtc_mean': rtc_mean,
            'rtc_std': rtc_std,
            'stc_mean': stc_mean,
            'stc_std': stc_std,
            'immune_mean': immune_mean,
            'immune_std': immune_std
        })

    pop_dynamics_RTC = pop_vs_time(results, colors=colors_t, pop="rtc", prefix="n_cells_i")
    pop_dynamics_RTC.savefig("plots/SP_impact_n_cells_i_population_RTC.png", dpi=150)
    
    pop_dynamics_STC = pop_vs_time(results, colors=colors_t, pop="stc", prefix="n_cells_i")
    pop_dynamics_STC.savefig("plots/SP_impact_n_cells_i_population_STC.png", dpi=150)

    pop_dynamics_Immune = pop_vs_time(results, colors=colors_i, pop="immune", prefix="n_cells_i")
    pop_dynamics_Immune.savefig("plots/SP_impact_n_cells_i_population_Immune.png", dpi=150)

    print("\n--- Résultats finaux des populations cellulaires ---")
    table_rows = []
    for row in final_counts:
        print(f"n_cells_i = {row['n_cells_i']}: RTC mean = {row['rtc_mean']:.2f} ± {row['rtc_std']:.2f}, STC mean = {row['stc_mean']:.2f} ± {row['stc_std']:.2f} + Immune mean = {row['immune_mean']:.2f} ± {row['immune_std']:.2f}")
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    df.to_csv("data/scenario_P_results.csv", index=False)
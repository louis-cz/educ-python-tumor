from simulation_tumor import simulation, pop_vs_time
import pandas as pd
import numpy as np


if __name__ == "__main__":
    
    sim_params = {
        'temps_cc': 24,
        'taille': 5,
        'n_jours': 350,
        'pmax': 10,
        'p_apoptose': 0.01,
        'pinit': 13,
    }

    img_intervals = [166, 333]
    n_sim_per_case = 3

    # 6 cas : 3 valeurs de mu x 2 valeurs de p_stc
    mu_values = [1, 5, 10]
    p_stc_values = [0.01, 0.1]  # 1% and 10%

    # on génère un dictionnaire imbriqué pour stocker les résultats
    # {1% : {mu1: [...], mu2: [...], ...}, 10%: {mu1: [...], mu2: [...], ...}}
    results = {f"p_stc={p_stc}": {f"mu={mu}": [] for mu in mu_values} for p_stc in p_stc_values}
    final_counts = [] # pour stocker les moyennes et écarts-types

    for p_stc in p_stc_values:
        sim_params['p_stc'] = p_stc

        for mu in mu_values:
            sim_params['mu'] = mu

            for sim_idx in range(n_sim_per_case):
                print(f"\n--- Simulation pour p_stc = {p_stc}, mu = {mu}, Simulation {sim_idx + 1}/{n_sim_per_case} ---")
                save_img = (sim_idx == 0)  # sauvegarder les images seulement pour la première simulation de chaque condition
                img_args = {
                    "save_img": save_img,
                    "img_itrvl": img_intervals if save_img else None,
                    "img_dir": f"img/S5_pstc_{int(p_stc*100)}_mu_{mu}" if save_img else None
                }
                cell_counts = simulation(sim_params, **img_args)
                results[f"p_stc={p_stc}"][f"mu={mu}"].append(cell_counts)

            # Calcul des moyennes et écarts-types des populations finales
            final_rtc_counts = [sim['rtc'][-1] for sim in results[f"p_stc={p_stc}"][f"mu={mu}"]]
            final_stc_counts = [sim['stc'][-1] for sim in results[f"p_stc={p_stc}"][f"mu={mu}"]]
            rtc_mean = np.mean(final_rtc_counts)
            rtc_std = np.std(final_rtc_counts)
            stc_mean = np.mean(final_stc_counts)
            stc_std = np.std(final_stc_counts)
            final_counts.append({
                'p_stc': p_stc,
                'mu': mu,
                'rtc_mean': rtc_mean,
                'rtc_std': rtc_std,
                'stc_mean': stc_mean,
                'stc_std': stc_std
            })

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # 4 plots à générer
        # 1. RTC population vs time Ps=1% et mu variable
    pop_dynamics = pop_vs_time(results["p_stc=0.01"], colors=colors, pop="rtc", prefix="mu")
    pop_dynamics.savefig("plots/S5_impact_mu_pstc_population_rtc_ps1.png", dpi=150)
        # 2. STC population vs time Ps=1% et mu variable
    pop_dynamics = pop_vs_time(results["p_stc=0.01"], colors=colors, pop="stc", prefix="mu")
    pop_dynamics.savefig("plots/S5_impact_mu_pstc_population_stc_ps1.png", dpi=150)
        # 3. RTC population vs time Ps=10% et mu variable
    pop_dynamics = pop_vs_time(results["p_stc=0.1"], colors=colors, pop="rtc", prefix="mu")
    pop_dynamics.savefig("plots/S5_impact_mu_pstc_population_rtc_ps10.png", dpi=150)
        # 4. STC population vs time Ps=10% et mu variable
    pop_dynamics = pop_vs_time(results["p_stc=0.1"], colors=colors, pop="stc", prefix="mu")
    pop_dynamics.savefig("plots/S5_impact_mu_pstc_population_stc_ps10.png", dpi=150)

    # Tableau rangé des résultats et sauvegarde dans un fichier

    print("\n--- Résultats finaux des populations cellulaires ---")
    table_rows = []
    for row in final_counts:
        print(f"{row['mu']}\t{int(row['p_stc']*100)}%\t{row['rtc_mean']:.0f} ± {row['rtc_std']:.0f}\t{row['stc_mean']:.1f} ± {row['stc_std']:.1f}")
        table_rows.append({
            "Migration pot.": row['mu'],
            "PS (%)": int(row['p_stc']*100),
            "Final RTC count": f"{row['rtc_mean']:.0f} ± {row['rtc_std']:.0f}",
            "Final STC count": f"{row['stc_mean']:.1f} ± {row['stc_std']:.1f}"
        })
    # enregistrement dans un fichier CSV
    df = pd.DataFrame(table_rows)
    df.to_csv("data/scenario_5_results.csv", index=False)

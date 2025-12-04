from simulation_tumor_immune import simulation, pop_vs_time

if __name__ == "__main__":

    # paramètres de la simulation
    sim_params = {
        # généraux
        'n_jours': 500,
        'taille': 50,
        # tumoraux
        "pmax_t": 10,
        "pinit_t": 13,
        "p_apoptose_t": 0.3,
        "p_proliferation_t": 0.1,
        "p_stc_t": 0.05,
        "mu_t": 5,
        # immunitaires
        "p_apoptose_i": 0.0,
        "p_proliferation_i": 0.025,
        "p_phagocytose_i": 0.5, # par jour
        "mu_i": 50,
        "pmax_i": -10,
        "n_cells_i": 50,
    }

    cell_counts = simulation(sim_params, show_anim=True)

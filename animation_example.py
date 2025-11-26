from simulation_tumor import simulation

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 30,
        'n_jours': 100,
        'p_stc' : 0.01,
        'mu': 10,
        'pmax': 10,
        'pinit': 12,
        'p_apoptose': 0.05
    }

    # changer is_at_end_of_day dans simulation_tumor.py si besoin
    cell_counts = simulation(sim_params, show_anim=True)
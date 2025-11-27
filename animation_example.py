from simulation_tumor import simulation

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

    # changer is_at_end_of_day dans simulation_tumor.py si besoin
    cell_counts = simulation(sim_params, show_anim=True)
from simulation_tumor import simulation

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        'temps_cc': 24,
        'taille': 10,
        'n_jours': 500,
        'p_stc' : 0.03,
        'mu': 10,
        'pmax': 10,
        'pinit': 13,
        'p_apoptose': 0.3,
    }

    # changer is_at_end_of_day dans simulation_tumor.py si besoin
    cell_counts = simulation(sim_params, save_img=True, img_dir="img/debug", img_itrvl=[10, 50, 100, 200, 300, 400, 499])

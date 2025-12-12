import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_tumor_immune import simulation
from PIL import Image


def to_gif(img_dir, gif_name, img_intervals):
    images = []
    for i in img_intervals:
        img_path = os.path.join(img_dir, f"simulation_jour_{i}.png")
        images.append(Image.open(img_path))
    
    images[0].save(
        gif_name,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

if __name__ == "__main__":
    
    # paramètres de la simulation
    sim_params = {
        # généraux
        'n_jours': 500,
        'taille': 100,
        # tumoraux
        "pmax_t": 15,
        "pinit_t": 18,
        "p_apoptose_t": 0.01,
        "p_proliferation_t": 0.1,
        "p_stc_t": 0.02,
        "mu_t": 5,
        # immunitaires
        "p_apoptose_i": 0.01,
        "p_proliferation_i": 0.03,
        "p_phagocytose_i": 0.5, # par jour
        "mu_i": 10,
        "n_cells_i" : 25
    }

    img_intervals = [i for i in range(sim_params['n_jours'])] 

    # sim
    cell_counts = simulation(sim_params, save_img=True, img_itrvl=img_intervals, img_dir="img/SP2_gif")

    # gif
    to_gif("img/SP2_gif", "SP2_simulation.gif", img_intervals[1:]) # remove day 0
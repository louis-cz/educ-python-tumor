from simulation_tumor import simulation
from PIL import Image
import os

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
        'temps_cc': 24,
        'taille': 200,
        'n_jours': 220,
        'p_apoptose': 0,
        'p_stc' : 0.05,
        'mu': 10,
        'pmax': 10,
        'pinit': 13
    }

    img_intervals = [i for i in range(sim_params['n_jours'])] 

    # sim
    cell_counts = simulation(sim_params, save_img=True, img_itrvl=img_intervals, img_dir="img/S3_gif")

    # gif
    to_gif("img/S3_gif", "S3_simulation.gif", img_intervals[1:]) # remove day 0
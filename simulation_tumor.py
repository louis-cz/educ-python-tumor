# 13-11-2025

# Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios
# Carlos A. Valentim, José A. Rabi, Sergio A. David

# bibliothèques
import sys  # gestion du système
import time  # gestion du temps
import os  # gestion des fichiers
import json
import numpy as np  # calculs numériques
import seaborn as sns  # visualisation 2D
import matplotlib.pyplot as plt  # affichage des graphiques
from scipy.ndimage import center_of_mass  # calcul du centre de masse
from matplotlib.colors import ListedColormap, BoundaryNorm # gestion des couleurs (heatmap)

# Description de la grille
# chaque case peut contenir une cellule (!= 0) ou être vide (0)
# la chiffre associée à chaque cellule indique son potentiel de prolifération
# à chaque division cellulaire, ce potentiel diminue de 1 jusqu'à atteindre 0
# une nouvelle STC est créée avec un potentiel de prolifération maximal + 1
# une nouvelle cellule clone est créée avec le même potentiel de prolifération que la cellule mère
# 0 : case vide

# utiliser matrix.nonzero() pour récupérer les coordonnées des cellules et diminuer le temps de calcul
# à côté on utilise un array pour stocker les coordonnées des cellules à mettre à jour
# plus facile que des for imbriquées

# pour les probabilités, on les génères en une fois pour toutes les cellules à chaque itération (car on sait le nombre de cellules)

# pour la migration, faire un shuffle des coordonnées des cellules avant de les traiter
# pour vérifier les voisins, ne pas oublier de faire un shuffle des coordonnées

# à chaque itération, il faut également recentrer la grille (le domaine de simulation) autour de la tumeur
# pour cela, on peut utiliser scipy.ndimage.center_of_mass pour trouver le centre de la tumeur

# == MÉTHODES POUR LA MISE À JOUR DE LA GRILLE == #

MOORE_COORDS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

    
def find_all_empty_neighbors(grille, coord_cells):
    """
    Trouve un voisin vide aléatoire pour chaque cellule dans coord_cells (ou -1 si aucun).
    grille : Grille de simulation.
    coord_cells : Coordonnées des cellules (array de forme (n_cells, 2)). 
    """
    n_cells = len(coord_cells)
    result = np.full((n_cells, 2), -1, dtype=int)  # init à -1 pour tout le monde
    
    moore_shuffled = MOORE_COORDS.copy() # copie pour ne pas modifier l'original
    np.random.shuffle(moore_shuffled) # shuffle les coordonées des voisins
    
    for dx, dy in moore_shuffled:
        # Calculer tous les voisins en une fois
        nx = coord_cells[:, 0] + dx # nouvelle coordonnée x (ligne → axe 0)
        ny = coord_cells[:, 1] + dy # nouvelle coordonnée y (colonne → axe 1)
        
        # check : bordures de la grille + case vide
        valid = (nx >= 0) & (nx < grille.shape[0]) & (ny >= 0) & (ny < grille.shape[1])
        empty = grille[nx[valid], ny[valid]] == 0
        
        # màj de résultat pour les cellules sans voisin trouvé
        valid_indices = np.where(valid)[0] # indices valides dans coord_cells
        empty_indices = valid_indices[empty] # indices des cellules avec voisin vide
        mask = result[empty_indices, 0] == -1  # Pas encore de voisin trouvé
        result[empty_indices[mask], 0] = nx[empty_indices[mask]] # mise à jour x
        result[empty_indices[mask], 1] = ny[empty_indices[mask]] # mise à jour y
    
    return result

# ici, on vérifie si une cellule est présente sur les bords de la grille
# pour ça on regarde les premières et dernières lignes et colonnes
# any() renvoie True si au moins un élément est vrai
def cellule_au_bord(grille):
    cond_l = grille[0, :].any() or grille[-1, :].any()
    cond_c = grille[:, 0].any() or grille[:, -1].any()
    return cond_l or cond_c
# == == #


# == FONCTIONS DE GESTION DE LA GRILLE == #

def grille_initialisation(taille, potentiel):
    grille = np.zeros((taille, taille), dtype="int")
    grille[taille // 2, taille // 2] = potentiel
    return grille


def grille_recentrage(grille):
    if not cellule_au_bord(grille):
        return grille  # pas besoin de recentrer si pas de cellule au bord
    else:
        non_zero = np.argwhere(grille > 0)
        if len(non_zero) == 0:
            return grille  # grille vide, pas de recentrage nécessaire
        
        min_x, min_y = non_zero.min(axis=0)
        max_x, max_y = non_zero.max(axis=0)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        taille = grille.shape[0]
        new_taille = taille + 2  # nouvelle taille double
        new_grille = np.zeros((new_taille, new_taille), dtype="int")

        offset_x = new_taille // 2 - center_x
        offset_y = new_taille // 2 - center_y

        # Ensure indices are within bounds
        x_start_new = max(offset_x, 0)
        y_start_new = max(offset_y, 0)
        x_end_new = min(offset_x + taille, new_taille)
        y_end_new = min(offset_y + taille, new_taille)

        x_start_old = max(-offset_x, 0)
        y_start_old = max(-offset_y, 0)
        x_end_old = x_start_old + (x_end_new - x_start_new)
        y_end_old = y_start_old + (y_end_new - y_start_new)

        new_grille[x_start_new:x_end_new, y_start_new:y_end_new] = grille[x_start_old:x_end_old, y_start_old:y_end_old]
        
        return new_grille


def grille_mise_a_jour(grille, p_apoptose, p_proliferation, p_stc, p_migration, pmax):

    coord_cells = np.transpose(np.nonzero(grille)) # liste des coordonnées des cellules
    n_cells = len(coord_cells)

    proba_apoptose = np.random.rand(n_cells)
    proba_proliferation = np.random.rand(n_cells)
    proba_migration = np.random.rand(n_cells)

    potentiels = grille[coord_cells[:, 0], coord_cells[:, 1]]
    is_true_stem = potentiels > pmax + 2
    is_clonogenic_stem = potentiels == pmax + 2
    is_rtc = potentiels < pmax + 2

    # grille_new = grille.copy()
    grille_new = grille.copy()
    voisins_vides = find_all_empty_neighbors(grille, coord_cells)

    # 1. Apoptose (vérifiée sur TOUTES les cellules)
    for cell_idx in range(n_cells):
        x, y = coord_cells[cell_idx]
        potentiel = potentiels[cell_idx]

        # Si la cellule a déjà été supprimée, on passe à la suivante
        if grille_new[x, y] == 0:
            continue

        # Apoptose (RTC uniquement)
        if is_rtc[cell_idx] and proba_apoptose[cell_idx] < p_apoptose:
            grille_new[x, y] = 0

    # 2. Prolifération et Migration (seulement pour les cellules avec voisin libre)
    has_voisin = voisins_vides[:, 0] != -1
    cellules_avec_voisin = np.where(has_voisin)[0]
    np.random.shuffle(cellules_avec_voisin)

    for cell_idx in cellules_avec_voisin:
        x, y = coord_cells[cell_idx]
        potentiel = potentiels[cell_idx]

        # Si la cellule a déjà été supprimée ou déplacée, on passe à la suivante
        if grille_new[x, y] == 0:
            continue

        # 2. Recherche de voisins vides pour prolifération ou migration
        voisin_vide = voisins_vides[cell_idx]
        if (voisin_vide == -1).all():
            continue  # pas de voisins vides, quiescence

        vx, vy = voisin_vide

        if grille_new[vx, vy] != 0:
            continue  # le voisin a déjà été occupé entre-temps

        # 3. Prolifération
        if proba_proliferation[cell_idx] < p_proliferation:
            if is_true_stem[cell_idx]:
                # True stem cell division: asymmetric (RTC) or symmetric (true stem)
                if np.random.random() < p_stc:
                    grille_new[vx, vy] = pmax + 3  # new true stem (symmetric division)
                else:
                    grille_new[vx, vy] = pmax + 1  # new RTC (asymmetric division)
            elif is_clonogenic_stem[cell_idx]:
                # Clonogenic stem cell division: only RTC daughters
                grille_new[vx, vy] = pmax + 1  # new RTC
            else:
                # RTC division: both mother and daughter lose 1 potential
                grille_new[vx, vy] = potentiel - 1
                grille_new[x, y] = potentiel - 1
            continue

        # 4. Migration
        elif proba_migration[cell_idx] < p_migration:
            grille_new[vx, vy] = potentiel
            grille_new[x, y] = 0

    return grille_new

# == == #


# == SIMULATION == #

def simulation(
    parameters: dict, 
    show_anim=False, 
    save_img=False, 
    img_itrvl=[], 
    img_dir="img", 
    save_json=False
):
    """
    Exécute la simulation de croissance tumorale avec les paramètres spécifiés. 
    ## Paramètres de la fonction :
        parameters  (dict)          : Dictionnaire contenant les paramètres de la simulation /!\
        show_anim   (bool = False)  : Si True, affiche l'animation en temps réel.
        save_img    (bool = False)  : Si True, sauvegarde les images de la simulation dans le dossier spécifié.
        img_itrvl   (list = [])     : Liste des jours pour sauvegarder les images.
        img_dir     (str = "img")   : Dossier où les images seront sauvegardées.
    ATTENTION ! Les paramètres de la simulation sont à définir dans un dictionnaire avec les clefs suivantes :
    ## Paramètres de la simulation :
        temps_cc    (int = 24)      : Durée du cycle cellulaire en heures.
        pmax        (int = 10)      : Potentiel de prolifération maximal pour une cellule stem-like (STC).
        pinit       (int = 12)      : Potentiel de prolifération initial des cellules au début de la simulation.
        taille      (int = 100)     : Taille de la grille (taille x taille).
        n_jours     (int = 50)      : Durée de la simulation en jours.
        p_apoptose  (float = 0)     : Probabilité d'apoptose par jour.
        p_stc       (float = 0.1)   : Probabilité de transition de cellule stem-like à cellule différenciée par prolifération.
        mu          (float = 10)    : Facteur de migration par jour.
    """

    # == 1. Extraction des paramètres == #

    # valeurs par défaut
    default_values = {
        "temps_cc": 24,
        "pmax": 10,
        "pinit": 12,
        "taille": 100,
        "n_jours": 50,
        "p_apoptose": 0.0,
        "p_stc": 0.1,
        "mu": 10,
    }

    # paramètres avec valeurs par défaut
    temps_cc = parameters.get("temps_cc", default_values["temps_cc"])
    pmax = parameters.get("pmax", default_values["pmax"])
    pinit = parameters.get("pinit", default_values["pinit"])
    taille = parameters.get("taille", default_values["taille"])
    n_jours = parameters.get("n_jours", default_values["n_jours"])
    p_apoptose = parameters.get("p_apoptose", default_values["p_apoptose"])
    p_stc = parameters.get("p_stc", default_values["p_stc"])
    mu = parameters.get("mu", default_values["mu"])

    # paramètres supplémentaires
    dt = 1 / 24  # Time step (1 hour = 1/24 day)
    p_proliferation = 24 / temps_cc * dt
    p_migration = mu * dt
    n_steps = int(n_jours / dt)  # nombre total de pas horaires
    p_apoptose_dt = p_apoptose * dt  # probabilité d'apoptose par pas de temps

    # == 2. Gestion dossier si save_img est True == #

    if save_img:
        if not os.path.exists(img_dir):  # si on ne trouve pas le dossier
            os.makedirs(img_dir)  # on le crée
        # supprimer les anciennes images
        for filename in os.listdir(img_dir):
            if filename.startswith("simulation_jour_") and filename.endswith(".png"):
                file_path = os.path.join(img_dir, filename)
                os.remove(file_path)

    # == 3. Simulation == #

    grille = grille_initialisation(taille, pinit)
    cell_counts = {
        "total": [grille.nonzero()[0].size],
        "rtc": [grille[(grille > 0) & (grille <= pmax + 1)].nonzero()[0].size],
        "stc": [grille[grille > pmax + 1].nonzero()[0].size],
    }

    if show_anim or save_img:
        _, ax = plt.subplots()

    start_time = time.time()
    print_freq = max(1, n_steps // 100)

    # On ne sauvegarde qu'une grille par jour pour les résultats
    heures_par_jour = int(1 / dt)

    for step in range(n_steps):  # boucle principale de la simulation (pas horaires)
        grille = grille_mise_a_jour(grille, p_apoptose_dt, p_proliferation, p_stc, p_migration, pmax)
        grille = grille_recentrage(grille)

        # Sauvegarde de la grille à la fin de chaque jour
        if (step + 1) % heures_par_jour == 0 or step == n_steps - 1:
            cell_counts["total"].append(grille.nonzero()[0].size)
            cell_counts["rtc"].append(grille[(grille > 0) & (grille <= pmax + 1)].nonzero()[0].size)
            cell_counts["stc"].append(grille[grille > pmax + 1].nonzero()[0].size)

        # affichage progression
        if step % print_freq == 0 or step == n_steps - 1:
            elapsed = time.time() - start_time
            steps_left = n_steps - (step + 1)
            days_left = steps_left * dt
            eta = elapsed / (step + 1) * steps_left if step > 0 else 0
            sys.stdout.write(
                f"\rHeure {step + 1}/{n_steps} "
                f"({days_left:.2f} jours restants) | "
                f"({(step + 1) * 100 // n_steps}%) | "
                f"Temps écoulé : {elapsed:.1f}s | "
                f"Temps restant : {eta:.1f}s"
            )
            sys.stdout.flush()  # permet de supprimer l'affichage précédent

        
        # condition :
        # 1. fin de journée et show_anim = True
        # 2. fin de journée et save_img = True et jour dans img_itrvl
        is_at_end_of_day = (step + 1) % heures_par_jour == 0
        # is_at_end_of_day = True
        if (is_at_end_of_day and (show_anim or (save_img and (((step + 1) // heures_par_jour) in img_itrvl)))):
            ax.clear()
            # si 0 → blanc, si pmax + 2 → jaune foncé (STC like), si pmax + 3 → jaune clair (true STC)
            # sinon (de 1 à pmax + 1), les RTCs ont une couleur entre noir et rouge selon leur potentiel
            colors = [(1, 1, 1)]  # blanc
            for i in range(1, pmax + 2):
                colors.append((i / (pmax + 1), 0, 0))  # dégradé de rouge
            colors.append((1, 1, 0))  # jaune foncé (STC like)
            colors.append((1, 1, 0))  # jaune foncé (true STC)

            # Vérification de la longueur de colors et bounds (copilot)
            bounds = list(range(len(colors)))  # bornes pour chaque couleur (0 à len(colors)-1)
            cmap = ListedColormap(colors)  # création de la colormap
            norm = BoundaryNorm(bounds, cmap.N)  # normalisation des couleurs
            #

            sns.heatmap(
                grille, cmap=cmap, cbar=False, ax=ax, norm=norm, vmin = 0, vmax = pinit
            )
            jour_actuel = (step + 1) // heures_par_jour
            ax.set_title(f"Jour {jour_actuel}", fontsize=14)
            ax.set_xlabel(
                f"Taille: {taille}x{taille} | Cycle cellulaire: {temps_cc}h\n"
                f"Temps simulé: {n_jours}j | Pas de temps: {dt*24:.0f}h\n"
                f"Pp: {p_proliferation:.3f} | Ps: {p_stc:.3f} | "
                f"mu: {mu} | Pa: {p_apoptose:.3f}\n"
                f"Pmax: {pmax}",
                fontsize=10,
            )
            plt.tight_layout()
            if save_img:
                plt.savefig(
                    os.path.join(img_dir, f"simulation_jour_{jour_actuel}.png"), dpi=150
                )
            if show_anim:
                plt.draw()
                plt.pause(0.05)

    print()
    if show_anim:
        plt.show()

    if save_json:
        suffix = f"{time.time():.0f}"
        results = {
            "parameters": parameters,
            "cell_counts": cell_counts,
        }
        # Ensure the 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(f"data/simulation_{suffix}.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    return cell_counts


# == == #

# == PLOTTING DES RÉSULTATS == #

def pop_vs_time(
    results,
    colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    log_scale=False,
    pop="total",
    prefix="",
):
    conditions = results.keys()  # list of tested conditions

    plt.figure(figsize=(8, 6))

    for idx, condition in enumerate(conditions):
        all_cell_counts = results[condition]  # list of cell_counts dicts

        # choix du type de population à tracer
        if pop == "total":
            populations = [
                np.array(sim_result["total"]) for sim_result in all_cell_counts
            ]
        elif pop == "stc":
            populations = [
                np.array(sim_result["stc"]) for sim_result in all_cell_counts
            ]
        elif pop == "rtc":
            populations = [
                np.array(sim_result["rtc"]) for sim_result in all_cell_counts
            ]

        # paramètres du plot
        n_days = min(len(pop) for pop in populations)
        populations_arr = np.array([pop[:n_days] for pop in populations])
        mean_pop = populations_arr.mean(axis=0)  # moyenne
        std_pop = populations_arr.std(axis=0)  # écart-type
        x = np.arange(1, n_days + 1)  # temps en jours
        color = colors[idx % len(colors)]  # couleur pour la condition actuelle

        plt.plot(
            x,
            mean_pop,
            label=f"{prefix} = {condition}",
            color=color,
            linewidth=2,
        )
        plt.fill_between(
            x, 
            mean_pop - std_pop, 
            mean_pop + std_pop, 
            alpha=0.3, color=color
        )

    if log_scale:
        plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1)
    plt.ylim(bottom=1)
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("Cell count", fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()

    return plt
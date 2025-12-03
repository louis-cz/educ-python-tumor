# 13-11-2025

# Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios
# Carlos A. Valentim, José A. Rabi, Sergio A. David

# bibliothèques
import sys  # gestion du système
import time  # gestion du temps
import os  # gestion des fichiers
import json # sauvegarde des résultats
import numpy as np  # calculs numériques
import seaborn as sns  # visualisation 2D
import matplotlib.pyplot as plt  # affichage des graphiques
from matplotlib.colors import ListedColormap, BoundaryNorm # gestion des couleurs (heatmap)


# Ici, on intègre des cellules immunitaires dans le modèle de croissance tumorale.
# représentée par un potentiel croissant négatif (-pmax_immune → 0)
# de couleur vert clair à vert foncé.


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
    Trouve un voisin vide aléatoire pour chaque cellule dans coord_cells (ou -1 si aucun)
    """
    n_cells = len(coord_cells) # nombre de cellules à traiter
    result = np.full((n_cells, 2), -1, dtype=int)  # init à -1 pour tout le monde
    
    moore_shuffled = MOORE_COORDS.copy() # copie pour ne pas modifier l'original
    np.random.shuffle(moore_shuffled) # shuffle les coordonées des voisins
    
    for dx, dy in moore_shuffled: # pour chaque coordonnée de voisin
        nx = coord_cells[:, 0] + dx # [:, 0] car x = ligne → axe 0
        ny = coord_cells[:, 1] + dy # [:, 1] car y = colonne → axe 1
        
        # check : bordures de la grille + case vide
        valid = (nx >= 0) & (nx < grille.shape[0]) & (ny >= 0) & (ny < grille.shape[1])
        empty = grille[nx[valid], ny[valid]] == 0
        
        # màj de résultat pour les cellules sans voisin trouvé
        valid_indices = np.where(valid)[0] # filtre aux indices valides dans coord_cells
        empty_indices = valid_indices[empty] # filtre aux indices des cellules avec voisin vide
        mask = result[empty_indices, 0] == -1  # Pas encore de voisin trouvé
        result[empty_indices[mask], 0] = nx[empty_indices[mask]] # mise à jour x
        result[empty_indices[mask], 1] = ny[empty_indices[mask]] # mise à jour y
    
    # result est donc sous la forme [[x1, y1], [x2, y2], ...] ou [-1, -1] si pas de voisin vide
    return result


def cellule_au_bord(grille):
    """
    Vérifie si une cellule est présente sur les bords de la grille.
    Pour ça on regarde les premières et dernières lignes et colonnes
    """
    # any() renvoie True si au moins un élément est vrai
    cond_l = grille[0, :].any() or grille[-1, :].any() # [0, :] → première ligne et [-1, :] → dernière ligne
    cond_c = grille[:, 0].any() or grille[:, -1].any() # [:, 0] → première colonne et [:, -1] → dernière colonne
    return cond_l or cond_c

# == == #



# == FONCTIONS DE GESTION DE LA GRILLE == #

def grille_initialisation(taille):
    grille = np.zeros((taille, taille), dtype="int")  # grille vide
    return grille

def grille_initialisation_tumor(grille, potentiel_tumor):
    center = grille.shape[0] // 2
    grille[center, center] = potentiel_tumor
    return grille

def grille_initialisation_immune(grille, n_cells, pmax_i):
    taille = grille.shape[0]
    placed = 0
    while placed < n_cells:
        x = np.random.randint(0, taille)
        y = np.random.randint(0, taille)
        if grille[x, y] == 0:  # case vide
            grille[x, y] = pmax_i
            placed += 1
    return grille

def grille_recentrage(grille):
    """
    Recentre la grille autour de la tumeur si une cellule est au bord
    """
    if not cellule_au_bord(grille):
        return grille  # pas besoin de recentrer si pas de cellule au bord
    else:
        non_zero = np.argwhere(grille != 0) # on récupère les coordonnées des cellules
       
        # on trouve les min et max en x et y
        min_x, min_y = non_zero.min(axis=0)
        max_x, max_y = non_zero.max(axis=0) 

        # équivalent de trouver le centre de masse
        center_x = (min_x + max_x) // 2 
        center_y = (min_y + max_y) // 2

        taille = grille.shape[0] 
        new_taille = taille + 2  # on ajoute deux marges
        new_grille = np.zeros((new_taille, new_taille), dtype="int") # nouvelle grille vide

        # calcul des décalages
        offset_x = new_taille // 2 - center_x
        offset_y = new_taille // 2 - center_y

        # /!\ → bords peuvent dépasser donc on réduit
        x_start_new = max(offset_x, 0)
        y_start_new = max(offset_y, 0)
        x_end_new = min(offset_x + taille, new_taille)
        y_end_new = min(offset_y + taille, new_taille)

        # et on calcule les indices correspondants dans l'ancienne grille
        x_start_old = max(-offset_x, 0)
        y_start_old = max(-offset_y, 0)
        x_end_old = x_start_old + (x_end_new - x_start_new)
        y_end_old = y_start_old + (y_end_new - y_start_new)

        # copie des valeurs de l'ancienne grille vers la nouvelle
        new_grille[x_start_new:x_end_new, y_start_new:y_end_new] = grille[x_start_old:x_end_old, y_start_old:y_end_old]
        
        return new_grille


def grille_mise_a_jour(grille, 
                       p_apoptose_t, p_proliferation_t, p_stc_t, p_migration_t, pmax_t,
                       p_apoptose_i, p_proliferation_i, p_migration_i, p_phagocytose_i, pmax_i):

    coord_cells = np.transpose(np.nonzero(grille)) # liste des coordonnées des cellules
    n_cells = len(coord_cells)

    proba_apoptose = np.random.rand(n_cells)
    proba_proliferation = np.random.rand(n_cells)
    proba_migration = np.random.rand(n_cells)

    potentiels = grille[coord_cells[:, 0], coord_cells[:, 1]]
    t_is_true_stem = potentiels > pmax_t + 2
    t_is_clonogenic_stem = potentiels == pmax_t + 2
    t_is_rtc = potentiels < pmax_t + 2
    i_is_immune = potentiels < 0

    # grille_new = grille.copy()
    grille_new = grille.copy()
    voisins_vides = find_all_empty_neighbors(grille, coord_cells)

    # 1. Apoptose et phagocytose (toutes les cellules)
    for cell_idx in range(n_cells):
        x, y = coord_cells[cell_idx]
        potentiel = potentiels[cell_idx]

        # Si la cellule a déjà été supprimée, on passe à la suivante
        if grille_new[x, y] == 0:
            continue

        # Apoptoses tumorales
        if t_is_rtc[cell_idx] and proba_apoptose[cell_idx] < p_apoptose_t:
            grille_new[x, y] = 0

        # Apoptoses immunitaires
        elif i_is_immune[cell_idx] and proba_apoptose[cell_idx] < p_apoptose_i:
            grille_new[x, y] = 0

        # Phagocytose des cellules tumorales par les cellules immunitaires
        if i_is_immune[cell_idx]:
            voisin_vide = voisins_vides[cell_idx]
            for dx, dy in MOORE_COORDS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grille.shape[0] and 0 <= ny < grille.shape[1]:
                    if grille[nx, ny] > 0:  # cellule tumorale détectée
                        # Tentative de prolifération
                        if proba_proliferation[cell_idx] < p_proliferation_i:
                            if voisin_vide[0] != -1 and grille_new[voisin_vide[0], voisin_vide[1]] == 0:
                                grille_new[voisin_vide[0], voisin_vide[1]] = potentiel # prolif avec niveau actuel
                        # Tentative de phagocytose
                        if np.random.rand() < p_phagocytose_i:
                            grille_new[nx, ny] = 0  # phagocytose
                            grille_new[x, y] = potentiel + 1 # perte d'énergie
                        break  # une seule cible par cellule immunitaire

    # 2. Prolifération et Migration (seulement pour les cellules avec >=1 voisin libre)
    has_voisin = voisins_vides[:, 0] != -1 # on check seulement la première colonne
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
            continue  # pas de voisins vides → quiescence

        vx, vy = voisin_vide

        if grille_new[vx, vy] != 0:
            continue  # le voisin a déjà été occupé entre-temps → quiescence

        # 3. Prolifération tumorale
        if potentiel > 0 and proba_proliferation[cell_idx] < p_proliferation_t:
            if t_is_true_stem[cell_idx]:
                
                if np.random.random() < p_stc_t:
                    grille_new[vx, vy] = pmax_t + 3  # STC
                else:
                    grille_new[vx, vy] = pmax_t + 1  # RTC
            elif t_is_clonogenic_stem[cell_idx]:
                grille_new[vx, vy] = pmax_t + 1  # RTC
            else:
                grille_new[vx, vy] = potentiel - 1 # RTC clone
                grille_new[x, y] = potentiel - 1 # mise à jour de la cellule mère
            continue

        # 4. Migration
        if proba_migration[cell_idx] < (p_migration_t if potentiel > 0 else p_migration_i):
            grille_new[vx, vy] = potentiel  # même potentiel
            grille_new[x, y] = 0  # case mère devient vide

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
    Exécute la simulation de croissance tumorale avec cellules immunitaires.
    ## Paramètres de la fonction :
        parameters  (dict)          : Dictionnaire contenant les paramètres de la simulation /!\
        show_anim   (bool = False)  : Si True, affiche l'animation en temps réel.
        save_img    (bool = False)  : Si True, sauvegarde les images de la simulation dans le dossier spécifié.
        img_itrvl   (list = [])     : Liste des jours pour sauvegarder les images.
        img_dir     (str = "img")   : Dossier où les images seront sauvegardées.
        save_json   (bool = False)  : Si True, sauvegarde les résultats en JSON.
    ATTENTION ! Les paramètres de la simulation sont à définir dans un dictionnaire avec les clefs suivantes :
    ## Paramètres généraux :
        temps_cc    (int = 24)      : Durée du cycle cellulaire en heures.
        taille      (int = 100)     : Taille de la grille (taille x taille).
        n_jours     (int = 50)      : Durée de la simulation en jours.
    ## Paramètres tumoraux :
        pmax_t      (int = 10)      : Potentiel de prolifération maximal pour une cellule stem-like (STC).
        pinit_t     (int = 12)      : Potentiel de prolifération initial des cellules tumorales.
        p_apoptose_t(float = 0.0)   : Probabilité d'apoptose tumorale par jour.
        p_stc_t     (float = 0.1)   : Probabilité de transition de cellule stem-like à cellule différenciée.
        mu_t        (float = 10)    : Facteur de migration tumorale par jour.
    ## Paramètres immunitaires :
        pmax_i      (int = -5)      : Potentiel des cellules immunitaires (négatif).
        pinit_i     (int = -5)      : Potentiel initial des cellules immunitaires.
        p_apoptose_i(float = 0.0)   : Probabilité d'apoptose immunitaire par jour.
        mu_i        (float = 10)    : Facteur de migration immunitaire par jour.
    """

    # == 1. Extraction des paramètres == #

    # valeurs par défaut
    default_values = {
        # général
        "temps_cc": 24,
        "taille": 100,
        "n_jours": 50,
        # tumorale
        "pmax_t": 10,
        "pinit_t": 12,
        "p_apoptose_t": 0.0,
        "p_proliferation_t": 0.0,
        "p_stc_t": 0.1,
        "mu_t": 10,
        # immunitaire
        "pmax_i": -5,
        "p_apoptose_i": 0.0,
        "p_proliferation_i": 0.0,
        "p_phagocytose_i": 0.0,
        "mu_i": 10,
        "n_cells_i": 10
    }

    # paramètres avec valeurs par défaut
    temps_cc = parameters.get("temps_cc", default_values["temps_cc"])
    taille = parameters.get("taille", default_values["taille"])
    n_jours = parameters.get("n_jours", default_values["n_jours"])
    # tumoraux
    pmax_t = parameters.get("pmax_t", default_values["pmax_t"])
    pinit_t = parameters.get("pinit_t", default_values["pinit_t"])
    p_apoptose_t = parameters.get("p_apoptose_t", default_values["p_apoptose_t"])
    p_proliferation_t = parameters.get("p_proliferation_t", default_values["p_proliferation_t"])
    p_stc_t = parameters.get("p_stc_t", default_values["p_stc_t"])
    mu_t = parameters.get("mu_t", default_values["mu_t"])
    # immunitaires
    pmax_i = parameters.get("pmax_i", default_values["pmax_i"])
    p_apoptose_i = parameters.get("p_apoptose_i", default_values["p_apoptose_i"])
    p_proliferation_i = parameters.get("p_proliferation_i", default_values["p_proliferation_i"])
    p_phagocytose_i = parameters.get("p_phagocytose_i", default_values["p_phagocytose_i"])
    mu_i = parameters.get("mu_i", default_values["mu_i"])
    n_cells_i = parameters.get("n_cells_i", default_values["n_cells_i"])

    # paramètres supplémentaires
    dt = 1 / 24  # Time step (1 hour = 1/24 day)
    n_steps = int(n_jours / dt)  # nombre total de pas horaires
    p_proliferation_t_dt = p_proliferation_t  # probabilité de prolifération tumorale par pas de temps
    p_proliferation_i_dt = p_proliferation_i  # probabilité de prolifération immunitaire par pas de temps
    p_migration_t = mu_t * dt  # probabilité de migration tumorale par pas de temps
    p_migration_i = mu_i * dt  # probabilité de migration immunitaire par pas de temps
    p_apoptose_t_dt = p_apoptose_t * dt  # probabilité d'apoptose tumorale par pas de temps
    p_apoptose_i_dt = p_apoptose_i * dt  # probabilité d'apoptose immunitaire par pas de temps
    p_phagocytose_i_dt = p_phagocytose_i * dt  # probabilité de phagocytose par pas de temps


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

    grille = grille_initialisation(taille)
    grille = grille_initialisation_tumor(grille, pinit_t)
    grille = grille_initialisation_immune(grille, n_cells_i, pmax_i)

    cell_counts = {
        "total": [grille.nonzero()[0].size],
        "rtc": [grille[(grille > 0) & (grille <= pmax_t + 1)].nonzero()[0].size],
        "stc": [grille[grille > pmax_t + 1].nonzero()[0].size],
        "immune": [grille[grille < 0].nonzero()[0].size],
    }

    if show_anim or save_img:
        _, ax = plt.subplots()

    start_time = time.time()
    print_freq = max(1, n_steps // 100)

    # On ne sauvegarde qu'une grille par jour pour les résultats
    heures_par_jour = int(1 / dt)

    for step in range(n_steps):  # boucle principale de la simulation (pas horaires)
        grille = grille_mise_a_jour(
            grille,
            p_apoptose_t_dt, p_proliferation_t_dt, p_stc_t, p_migration_t, pmax_t,
            p_apoptose_i_dt, p_proliferation_i_dt, p_migration_i, p_phagocytose_i_dt, pmax_i
        )
        grille = grille_recentrage(grille)

        # Sauvegarde de la grille à la fin de chaque jour
        if (step + 1) % heures_par_jour == 0 or step == n_steps - 1:
            cell_counts["total"].append(grille.nonzero()[0].size)
            cell_counts["rtc"].append(grille[(grille > 0) & (grille <= pmax_t + 1)].nonzero()[0].size)
            cell_counts["stc"].append(grille[grille > pmax_t + 1].nonzero()[0].size)
            cell_counts["immune"].append(grille[grille < 0].nonzero()[0].size)

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
            
            # pmax_i → -1 : degradé de vert clair à vert foncé
            # 0 : blanc
            # 1 à pmax_t +1 : dégradé de rouge clair à rouge foncé
            # pmax_t +2 et +3 : jaune et jaune
            colors = []
            for i in range(pmax_i, 0):
                green_value = int(255 - (i - pmax_i) * (200 / (0 - pmax_i)))
                colors.append(f"#00{green_value:02x}00")  # dégradé de vert
            colors.append("#ffffff")  # blanc pour 0
            for i in range(1, pmax_t + 2):
                red_value = int(255 - (i - 1) * (200 / (pmax_t + 1 - 1)))
                colors.append(f"#{red_value:02x}0000")  # dégradé de rouge
            colors.append("#ffff00")  # jaune pour pmax_t +2
            colors.append("#ffff00")  # jaune pour pmax_t +3

            # Vérification de la longueur de colors et bounds (copilot)
            bounds = list(range(pmax_i, pmax_t + 5))  # bornes de pmax_i à pmax_t + 4
            cmap = ListedColormap(colors)  # création de la colormap
            norm = BoundaryNorm(bounds, cmap.N)  # normalisation des couleurs
            #

            sns.heatmap(
                grille, cmap=cmap, cbar=False, ax=ax, norm=norm, vmin=pmax_i, vmax=pmax_t + 3
            )
            jour_actuel = (step + 1) // heures_par_jour
            ax.set_title(f"Jour {jour_actuel}", fontsize=14)
            # ax.set_xlabel(
            #     f"Taille: {taille}x{taille} | Cycle cellulaire: {temps_cc}h\n"
            #     f"Temps simulé: {n_jours}j | Pas de temps: {dt*24:.0f}h\n"
            #     f"Pp: {p_proliferation:.3f} | Ps: {p_stc:.3f} | "
            #     f"mu: {mu} | Pa: {p_apoptose:.3f}\n"
            #     f"Pmax: {pmax}",
            #     fontsize=10,
            # )
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
        elif pop == "immune":
            populations = [
                np.array(sim_result["immune"]) for sim_result in all_cell_counts
            ]

        # paramètres du plot
        n_days = min(len(p) for p in populations)
        populations_arr = np.array([p[:n_days] for p in populations])
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

# == == #
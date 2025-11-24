# 13-11-2025

# Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios
# Carlos A. Valentim, José A. Rabi, Sergio A. David

# bibliothèques
import sys  # gestion du système
import time  # gestion du temps
import os  # gestion des fichiers
import json
import numpy as np  # calculs numériques
import random as rd  # randomisation
import seaborn as sns  # visualisation 2D
import matplotlib.pyplot as plt  # affichage des graphiques
from scipy.ndimage import center_of_mass  # calcul du centre de masse
from matplotlib.colors import (
    ListedColormap,
    BoundaryNorm,
)  # gestion des couleurs (heatmap)

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


def get_coordonnees_voisins(x, y, taille_grille):
    """
    Renvoie les coordonnées des voisins de Moore pour une cellule (x, y).
    8 voisins possibles : NW, N, NE, W, E, SW, S, SE + gestion des bords de la grille incluse.
    """
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
    voisins = []
    for dx, dy in MOORE_COORDS:
        vx, vy = x + dx, y + dy
        if 0 <= vx < taille_grille and 0 <= vy < taille_grille:
            voisins.append((vx, vy))
    return voisins


def get_voisins_vides(x, y, grille):
    """
    Renvoie les coordonnées des voisins vides (valeur 0) pour une cellule donnée (x, y).
    """
    voisins = get_coordonnees_voisins(x, y, grille.shape[0])
    voisins_vides = [(vx, vy) for vx, vy in voisins if grille[vx, vy] == 0]
    return voisins_vides


def cellule_au_bord(grille):
    """
    Vérifie si une cellule est au bord de la grille (1ère colonne + dernière colonne + 1ère ligne + dernière ligne).
    1. on génère les coordonnées des bords
    2. on récupères les coordonnées des cellules non vides
    3. si une coordonnée de bord est dans les coordonnées des cellules, on retourne True
    """
    taille = grille.shape[0]
    bords_coords = []

    # 1ère et dernière ligne
    for j in range(taille):
        bords_coords.append((0, j))  # 1ère ligne
        bords_coords.append((taille - 1, j))  # dernière ligne

    # 1ère et dernière colonne
    for i in range(taille):
        bords_coords.append((i, 0))  # 1ère colonne
        bords_coords.append((i, taille - 1))  # dernière colonne

    # extraction des coordonnées des cellules
    coord_cells = np.nonzero(grille)
    coord_cells_set = set(
        zip(coord_cells[0], coord_cells[1])
    )  # transformation pour le set
    for coord in bords_coords:
        if coord in coord_cells_set:
            return True
    return False


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
        # on fait une nouvelle matrice en ajoutant des marges autour de l'ancienne
        taille = grille.shape[0]
        new_taille = taille + 2  # on ajoute une marge de 1 de chaque côté
        new_grille = np.zeros((new_taille, new_taille), dtype="int")
        # copier l'ancienne grille dans la nouvelle en la recentrant
        com = center_of_mass(grille)  # centre de masse de la tumeur
        com_x, com_y = int(com[0]), int(
            com[1]
        )  # case la plus proche du centre de masse
        start_x = max(
            0, min(new_taille - taille, new_taille // 2 - com_x)
        )  # position de départ pour copier l'ancienne grille (x)
        start_y = max(
            0, min(new_taille - taille, new_taille // 2 - com_y)
        )  # position de départ pour copier l'ancienne grille (y)
        new_grille[start_x : start_x + taille, start_y : start_y + taille] = grille

        return new_grille


def grille_mise_a_jour(grille, p_apoptose, p_proliferation, p_stc, p_migration, pmax):

    coord_cells = np.transpose(np.nonzero(grille))
    n_cells = len(coord_cells)
    order = np.arange(n_cells)
    np.random.shuffle(order)

    proba_apoptose = np.random.rand(n_cells)
    proba_proliferation = np.random.rand(n_cells)
    proba_migration = np.random.rand(n_cells)

    grille_new = grille.copy()

    for cell_idx in order:
        x, y = coord_cells[cell_idx]
        potentiel = grille_new[x, y]

        # Si la cellule a déjà été supprimée ou déplacée, on passe à la suivante
        if grille_new[x, y] == 0:
            continue

        # True stem cell: potentiel > pmax + 2
        is_true_stem = potentiel > pmax + 2
        # Clonogenic stem cell: potentiel == pmax + 2
        is_clonogenic_stem = potentiel == pmax + 2
        # RTC: potentiel <= pmax + 1
        is_rtc = potentiel <= pmax + 1

        # 1. Apoptose (RTC uniquement)
        if is_rtc and proba_apoptose[cell_idx] < p_apoptose:
            grille_new[x, y] = 0
            continue

        # 2. Recherche de voisins vides pour prolifération ou migration
        voisins_vides = get_voisins_vides(x, y, grille_new)
        if not voisins_vides:
            continue  # pas de voisins vides, quiescence

        np.random.shuffle(voisins_vides)
        vx, vy = voisins_vides[0]

        # 3. Prolifération
        if proba_proliferation[cell_idx] < p_proliferation:
            if is_true_stem:
                # True stem cell division: asymmetric (RTC) or symmetric (true stem)
                if np.random.random() < p_stc:
                    grille_new[vx, vy] = (
                        pmax + 3
                    )  # new true stem cell (symmetric division)
                else:
                    grille_new[vx, vy] = pmax + 1  # new RTC (asymmetric division)
            elif is_clonogenic_stem:
                # Clonogenic stem cell division: only RTC daughters
                grille_new[vx, vy] = pmax + 1  # new RTC
            else:
                # RTC division: both mother and daughter lose 1 potential
                grille_new[vx, vy] = potentiel - 1
                grille_new[x, y] = potentiel - 1

        # 4. Migration
        elif proba_migration[cell_idx] < p_migration:
            grille_new[vx, vy] = potentiel
            grille_new[x, y] = 0

        # 5. Quiescence (aucune action possible si aucune condition n'est remplie)

    return grille_new


# == == #


# == SIMULATION == #


def simulation(
    parameters: dict, show_anim=False, save_img=False, img_itrvl=[], img_dir="img"
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
    p_apoptose = p_apoptose * dt  # probabilité d'apoptose par pas de temps

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

    grilles = []
    grille = grille_initialisation(taille, pinit)
    grilles.append(grille.copy())

    if show_anim or save_img:
        _, ax = plt.subplots()

    start_time = time.time()
    print_freq = max(1, n_steps // 100)

    # On ne sauvegarde qu'une grille par jour pour les résultats
    grilles_par_jour = [grille.copy()]
    heures_par_jour = int(1 / dt)

    for step in range(n_steps):  # boucle principale de la simulation (pas horaires)
        grille = grille_mise_a_jour(
            grille, p_apoptose, p_proliferation, p_stc, p_migration, pmax
        )
        grille = grille_recentrage(grille)

        # Sauvegarde de la grille à la fin de chaque jour
        if (step + 1) % heures_par_jour == 0 or step == n_steps - 1:
            grilles_par_jour.append(grille.copy())

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

        # affichage / sauvegarde image (si le jour correspond est dans img_itrvl)
        if (show_anim or save_img) and ((step + 1) // heures_par_jour in img_itrvl):
            ax.clear()
            # si 0 → blanc, si pmax + 1 → jaune (STC)
            # sinon (de 1 à pmax + 1), les RTCs ont une couleur entre noir et rouge selon leur potentiel
            colors = [(1, 1, 1)]  # blanc
            for i in range(1, pmax + 1):
                frac = i / pmax
                colors.append((frac, 0, 0))  # RGB: noir→rouge
            colors.append((1, 1, 0))  # jaune pour STC clonogenic (pmax + 2)
            colors.append((1, 1, 0))  # jaune pour STC true stem (pmax + 3)

            # généré par copilot #
            cmap = ListedColormap(colors)  # création de la colormap
            bounds = list(
                range(pmax + 4)
            )  # bornes pour chaque couleur (0 à pmax+3 inclus)
            norm = BoundaryNorm(bounds, cmap.N)  # normalisation des couleurs
            #

            sns.heatmap(
                grille, cmap=cmap, cbar=False, ax=ax, vmin=0, vmax=pmax + 3, norm=norm
            )
            jour_actuel = (step + 1) // heures_par_jour
            ax.set_title(f"Jour {jour_actuel}", fontsize=14)
            ax.set_xlabel(
                f"Taille: {taille}x{taille} | Cycle cellulaire: {temps_cc}h\n"
                f"Temps simulé: {n_jours}j | Pas de temps: {dt*24:.0f}h\n"
                f"Pp: {p_proliferation:.2f} | Ps: {p_stc:.2f} | "
                f"Pm: {p_migration:.2f} | Pa: {p_apoptose:.2f}\n"
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

    json_file = f"./data/sim_{int(time.time())}.json"
    grilles_par_jour_serializable = [grille.tolist() for grille in grilles_par_jour]
    with open(json_file, "w") as f:
        json.dump(grilles_par_jour_serializable, f)
    return grilles_par_jour


# == == #

# == PLOTTING DES RÉSULTATS == #


def pop_vs_time(
    results,
    conditions,
    conditions_val,
    colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    log_scale=False,
    pop="total",
):

    plt.figure(figsize=(8, 6))

    for idx, condition in enumerate(conditions):

        sim_grilles = results[
            condition
        ]  # Chaque sim_grilles = liste de listes de grilles à différents steps

        # choix du type de population à tracer
        if pop == "total":
            populations = [
                np.array([np.count_nonzero(grille) for grille in grilles])
                for grilles in sim_grilles
            ]
        elif pop == "stc":
            populations = [
                np.array(
                    [
                        np.count_nonzero(grille >= conditions_val[idx] + 2)
                        for grille in grilles
                    ]
                )
                for grilles in sim_grilles
            ]
        elif pop == "rtc":
            populations = [
                np.array(
                    [
                        np.count_nonzero(grille <= conditions_val[idx] + 1)
                        for grille in grilles
                    ]
                )
                for grilles in sim_grilles
            ]

        # paramètres du plot
        min_len = min([len(pop) for pop in populations])  # on prend la plus petite sim
        pops_arr = np.array(
            [pop[:min_len] for pop in populations]
        )  # on aligne les tailles
        mean_pop = pops_arr.mean(axis=0)  # moyenne
        std_pop = pops_arr.std(axis=0)  # écart-type
        x = np.arange(1, min_len + 1)  # temps en jours
        color = colors[idx % len(colors)]  # couleur pour la condition actuelle

        plt.plot(
            # données à tracer (jours, population moyenne)
            x,
            mean_pop,
            label=f"pmax={conditions_val[idx]}",
            color=color,
            linewidth=2,
        )
        plt.fill_between(
            x, mean_pop - std_pop, mean_pop + std_pop, alpha=0.3, color=color
        )

    if log_scale:
        plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, None)
    plt.ylim(1, None)
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("Cell count", fontsize=13)
    plt.legend(fontsize=12)
    plt.tight_layout()
    return plt

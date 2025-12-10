import numpy as np
import time

def cellule_au_bord_before(grille):
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

def cellule_au_bord_after(grille):
    """
    Vérifie si une cellule est présente sur les bords de la grille.
    Pour ça on regarde les premières et dernières lignes et colonnes
    """
    # any() renvoie True si au moins un élément est vrai
    cond_l = grille[0, :].any() or grille[-1, :].any() # [0, :] → première ligne et [-1, :] → dernière ligne
    cond_c = grille[:, 0].any() or grille[:, -1].any() # [:, 0] → première colonne et [:, -1] → dernière colonne
    return cond_l or cond_c

if __name__ == "__main__":

    print("-- cellule_au_bord --")
    print("Test sur une grille 10000x10000 avec cellule au bord")

    # init
    taille = 10000
    grille = np.zeros((taille, taille), dtype=int)
    grille[0, taille//2] = 1  # cellule au bord

    # before
    start_time = time.time()
    result_before = cellule_au_bord_before(grille)
    end_time = time.time()
    print(f"BEFORE : Temps écoulé: {end_time - start_time:.6f} secondes")

    # after
    start_time = time.time()
    result_after = cellule_au_bord_after(grille)
    end_time = time.time()
    print(f"AFTER : Temps écoulé: {end_time - start_time:.6f} secondes")

    print("--")

    print("Test sur une grille 100x100 avec cellule au bord mais x10000")

    taille = 100
    grille = np.zeros((taille, taille), dtype=int)
    grille[0, taille//2] = 1  # cellule au bord
    steps = 10000

    # before
    start_time = time.time()
    for i in range(steps):
        result_before = cellule_au_bord_before(grille)
    end_time = time.time()
    print(f"BEFORE : Temps écoulé: {end_time - start_time:.6f} secondes")

        # after
    start_time = time.time()
    for i in range(steps):
        result_after = cellule_au_bord_after(grille)
    end_time = time.time()
    print(f"AFTER : Temps écoulé: {end_time - start_time:.6f} secondes")

    print("--")
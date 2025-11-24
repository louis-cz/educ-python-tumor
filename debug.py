from simulation_tumor import pop_vs_time
import json
import numpy as np


# TODO: corriger la pop_vs_time pour les RTC à partir du scénario 3 et 4

# 1. import des grilles
dir = "./data/"
papt_val = [0, 0.01, 0.1, 0.3]  # valeurs de papt à tester
file_path = {
    0: ["S4_papt0_1.json", "S4_papt0_2.json", "S4_papt0_3.json"],
    0.01: ["S4_papt0.01_1.json", "S4_papt0.01_2.json", "S4_papt0.01_3.json"],
    0.1: ["S4_papt0.1_1.json", "S4_papt0.1_2.json", "S4_papt0.1_3.json"],
    0.3: ["S4_papt0.3_1.json", "S4_papt0.3_2.json", "S4_papt0.3_3.json"],
}

data = {0: [], 0.01: [], 0.1: [], 0.3: []}
for papt in papt_val:
    for file_name in file_path[papt]:
        with open(dir + file_name, "r") as f:
            grilles_par_jour = json.load(f)
            # grilles_par_jour est une liste de grilles (matrices) pour chaque jour
            # On reconstitue chaque grille sous forme de numpy array si besoin
            grilles_par_jour_reconstituees = [np.array(grille) for grille in grilles_par_jour]
            data[papt].append(grilles_par_jour_reconstituees)

# print nb cellules tous les 10 jours pour vérifier
for papt in papt_val:
    print(f"\n--- p_apoptose = {papt} ---")
    for sim_idx, grilles in enumerate(data[papt]):
        print(f"Simulation {sim_idx + 1}:")
        for day in range(0, len(grilles), 10):
            grid = grilles[day]
            n_stc = np.sum(grid > 11)
            n_rtc = np.sum(grid <= 11)
            print(f" Day {day}: STC = {n_stc}, RTC = {n_rtc}")

# 2. on resimule le graph des RTC
pop_dynamics_RTC = pop_vs_time(
    data,
    papt_val,
    colors=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"],
    pmax=10,
    pop="rtc",
)
pop_dynamics_RTC.show()

pop_dynamics_STC = pop_vs_time(
    data,
    papt_val,
    colors=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"],
    pmax=10,
    pop="stc",
)
pop_dynamics_STC.show()
from simulation_tumor import pop_vs_time
import json


# TODO: corriger la pop_vs_time pour les RTC à partir du scénario 3 et 4

# 1. import des grilles
dir = "./data/"
pmax_val = [5, 10, 15, 20]  # valeurs de pmax à tester
file_path = {
    5: ["S3_pmax5_1.json", "S3_pmax5_2.json", "S3_pmax5_3.json"],
    10: ["S3_pmax10_1.json", "S3_pmax10_2.json", "S3_pmax10_3.json"],
    15: ["S3_pmax15_1.json", "S3_pmax15_2.json", "S3_pmax15_3.json"],
    20: ["S3_pmax20_1.json", "S3_pmax20_2.json", "S3_pmax20_3.json"],
}

data = {5: [], 10: [], 15: [], 20: []}
for pmax in pmax_val:
    for file_name in file_path[pmax]:
        with open(dir + file_name, "r") as f:
            grilles = json.load(f)
            # Ensure grilles are lists for serialization (if not already)
            if isinstance(grilles, list):
                grilles_par_jour_serializable = [
                    grille if isinstance(grille, list) else list(grille)
                    for grille in grilles
                ]
            else:
                grilles_par_jour_serializable = [list(grilles)]
            data[pmax].append(grilles_par_jour_serializable)

# 2. on resimule le graph des RTC
pop_dynamics_RTC = pop_vs_time(
    data,
    pmax_val,
    pmax_val,
    colors=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"],
    pop="rtc",
)
pop_dynamics_RTC.show()

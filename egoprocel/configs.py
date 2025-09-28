"""Configurations for the EgoProcel datasets"""

DATASETS = {
    "MECCANO": {
        "annotations": "MECCANO",
        "video": "MECCANO",
        "features": "MECCANO",
        "num_keysteps": 17,
    },
    "EPIC-Tent": {
        "annotations": "EPIC-Tents",
        "video": "EPIC-Tent",
        "features": "EPIC-Tent",
        "num_keysteps": 12,
    },
    "pc_assembly": {
        "annotations": "pc_assembly",
        "video": "PC/assembly/CFR/rescaled",
        "features": "PC",
        "num_keysteps": 9,
    },
    "pc_disassembly": {
        "annotations": "pc_disassembly",
        "video": "PC/disassembly/CFR/",
        "features": "PC",
        "num_keysteps": 9,
    },
    "EGTEA": {
        "features": "EGTEA",
        "tasks": [
            {"name": "BaconAndEggs", "num_keysteps": 11, "annotations": "EGTEA_Gaze+/BaconAndEggs", "video": "EGTEA/BaconAndEggs"},
            {"name": "Cheeseburger", "num_keysteps": 10, "annotations": "EGTEA_Gaze+/Cheeseburger", "video": "EGTEA/Cheeseburger"},
            {"name": "ContinentalBreakfast", "num_keysteps": 10, "annotations": "EGTEA_Gaze+/ContinentalBreakfast", "video": "EGTEA/ContinentalBreakfast"},
            {"name": "Pizza", "num_keysteps": 8, "annotations": "EGTEA_Gaze+/Pizza", "video": "EGTEA/Pizza"},
            {"name": "GreekSalad", "num_keysteps": 4, "annotations": "EGTEA_Gaze+/GreekSalad", "video": "EGTEA/GreekSalad"},
            {"name": "PastaSalad", "num_keysteps": 8, "annotations": "EGTEA_Gaze+/PastaSalad", "video": "EGTEA/PastaSalad"},
            {"name": "TurkeySandwich", "num_keysteps": 6, "annotations": "EGTEA_Gaze+/TurkeySandwich", "video": "EGTEA/TurkeySandwich"},
        ],
    },
    "CMU": {
        "features": "CMU_Kitchens",
        "tasks": [
            {"name": "Brownie", "num_keysteps": 9, "annotations": "CMU_Kitchens/Brownie/ego", "video": "CMU_Kitchens/Brownie/ego"},
            {"name": "Eggs", "num_keysteps": 8, "annotations": "CMU_Kitchens/Eggs/ego", "video": "CMU_Kitchens/Eggs/ego"},
            {"name": "Pizza", "num_keysteps": 5, "annotations": "CMU_Kitchens/Pizza/ego", "video": "CMU_Kitchens/Pizza/ego"},
            {"name": "Salad", "num_keysteps": 9, "annotations": "CMU_Kitchens/Salad/ego", "video": "CMU_Kitchens/Salad/ego"},
            {"name": "Sandwich", "num_keysteps": 4, "annotations": "CMU_Kitchens/Sandwich/ego", "video": "CMU_Kitchens/Sandwich/ego"},
        ],
    },
}

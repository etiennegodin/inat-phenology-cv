from __future__ import annotations

LABEL_MAPPING = {
    0: "Flowering",
    1: "Fruiting",
    2: "Flower_Budding",
}

CLASS_ORDER = ["Flowering", "Fruiting", "Flower_Budding"]

OBSERVATIONS_FIELDS = {
    "id": True,
    "photos": True,
}

from __future__ import annotations


GROWTH_STAGE_BY_MM_DD = {
    "05-08": "Bloom / early fruit set",
    "05-12": "Bloom to fruit set",
    "05-18": "Green fruit development",
    "05-22": "Fruit sizing / early ripening",
    "05-27": "Ripening / early harvest window",
    "05-29": "Harvest-time ripening",
}


GROWTH_STAGE_NOTES = {
    "Bloom / early fruit set": "Early season marks flag establishment and early resource-limitation areas for scouting.",
    "Bloom to fruit set": "Flowering and early fruit set are active; red low-vigour zones should be checked first.",
    "Green fruit development": "Canopy and fruit demand are rising; persistent low and medium zones become more operationally important.",
    "Fruit sizing / early ripening": "The crop is approaching ripening; expanding or persistent zones should be reviewed before harvest pressure peaks.",
    "Ripening / early harvest window": "Harvest-time scouting should focus on repeated low and medium zones while separating field-edge effects from production rows.",
    "Harvest-time ripening": "Late-season class maps support harvest review and next-season scouting priorities.",
}


def growth_stage_for_date(date: str) -> str:
    month_day = date[5:10]
    return GROWTH_STAGE_BY_MM_DD.get(month_day, "Growth stage review required")


def growth_stage_note(stage: str) -> str:
    return GROWTH_STAGE_NOTES.get(stage, "Confirm growth stage from field scouting notes.")

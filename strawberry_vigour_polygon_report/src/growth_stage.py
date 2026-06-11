from __future__ import annotations


GROWTH_STAGE_BY_MM_DD = {
    "04-20": "Vegetative establishment / pre-bloom",
    "04-24": "Pre-bloom canopy expansion",
    "04-29": "Early bloom onset",
    "05-01": "Early bloom",
    "05-08": "Bloom / early fruit set",
    "05-12": "Bloom to fruit set",
    "05-18": "Green fruit development",
    "05-22": "Fruit sizing / early ripening",
    "05-27": "Ripening / early harvest window",
    "05-29": "Harvest-time ripening",
    "06-04": "Late harvest / post-harvest review",
}


GROWTH_STAGE_NOTES = {
    "Vegetative establishment / pre-bloom": "Early canopy differences are stage-sensitive and should be read as establishment variability, not harvest-time vigour.",
    "Pre-bloom canopy expansion": "Canopy expansion is still uneven; concern-region percentages should be compared only with nearby pre-bloom dates.",
    "Early bloom onset": "Bloom is beginning; vigour mix is transitional and should not be trended directly against fruit-sizing or ripening dates.",
    "Early bloom": "Early bloom marks active canopy and flower development; interpret low and medium areas as scouting prompts, not final yield classes.",
    "Bloom / early fruit set": "Early season marks flag establishment and early resource-limitation areas for scouting.",
    "Bloom to fruit set": "Flowering and early fruit set are active; red low-vigour zones should be checked first.",
    "Green fruit development": "Canopy and fruit demand are rising; persistent low and medium zones become more operationally important.",
    "Fruit sizing / early ripening": "The crop is approaching ripening; expanding or persistent zones should be reviewed before harvest pressure peaks.",
    "Ripening / early harvest window": "Harvest-time scouting should focus on repeated low and medium zones while separating field-edge effects from production rows.",
    "Harvest-time ripening": "Late-season class maps support harvest review and next-season scouting priorities.",
    "Late harvest / post-harvest review": "Late-season review should focus on persistence, recovery, and expansion of marked concern areas.",
}


def growth_stage_for_date(date: str) -> str:
    month_day = date[5:10]
    return GROWTH_STAGE_BY_MM_DD.get(month_day, "Growth stage review required")


def growth_stage_note(stage: str) -> str:
    return GROWTH_STAGE_NOTES.get(stage, "Confirm growth stage from field scouting notes.")

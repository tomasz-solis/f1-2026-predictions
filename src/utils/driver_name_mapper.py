"""Maps driver names between FastF1 abbreviations and full names."""

import logging

logger = logging.getLogger(__name__)


class DriverNameMapper:
    """Maps driver names between FastF1 abbreviations and full names."""

    # 2026 F1 Grid mapping: abbreviation -> full last name
    DRIVER_MAP = {
        # Red Bull
        "VER": "Verstappen",
        "PER": "Perez",
        # McLaren
        "NOR": "Norris",
        "PIA": "Piastri",
        # Ferrari
        "LEC": "Leclerc",
        "HAM": "Hamilton",
        # Mercedes
        "RUS": "Russell",
        "ANT": "Antonelli",
        # Aston Martin
        "ALO": "Alonso",
        "STR": "Stroll",
        # Alpine
        "GAS": "Gasly",
        "COL": "Colapinto",
        "DOO": "Doohan",
        # Williams
        "ALB": "Albon",
        "SAI": "Sainz",
        # Haas
        "BEA": "Bearman",
        "OCO": "Ocon",
        # Sauber/Audi
        "HUL": "Hulkenberg",
        "BOR": "Bortoleto",
        # RB
        "LAW": "Lawson",
        "LIN": "Lindblad",
        "TSU": "Tsunoda",
        # Cadillac
        "BOT": "Bottas",
        # Red Bull
        "HAD": "Hadjar",
    }

    # Reverse mapping: full name -> abbreviation
    REVERSE_MAP = {v: k for k, v in DRIVER_MAP.items()}

    # Alternative spellings and formats
    NAME_VARIANTS = {
        "max verstappen": "VER",
        "verstappen": "VER",
        "sergio perez": "PER",
        "perez": "PER",
        "lando norris": "NOR",
        "norris": "NOR",
        "oscar piastri": "PIA",
        "piastri": "PIA",
        "charles leclerc": "LEC",
        "leclerc": "LEC",
        "lewis hamilton": "HAM",
        "hamilton": "HAM",
        "george russell": "RUS",
        "russell": "RUS",
        "kimi antonelli": "ANT",
        "antonelli": "ANT",
        "fernando alonso": "ALO",
        "alonso": "ALO",
        "lance stroll": "STR",
        "stroll": "STR",
        "pierre gasly": "GAS",
        "gasly": "GAS",
        "franco colapinto": "COL",
        "colapinto": "COL",
        "jack doohan": "DOO",
        "doohan": "DOO",
        "alex albon": "ALB",
        "albon": "ALB",
        "carlos sainz": "SAI",
        "sainz": "SAI",
        "oliver bearman": "BEA",
        "bearman": "BEA",
        "esteban ocon": "OCO",
        "ocon": "OCO",
        "nico hulkenberg": "HUL",
        "hulkenberg": "HUL",
        "gabriel bortoleto": "BOR",
        "bortoleto": "BOR",
        "liam lawson": "LAW",
        "lawson": "LAW",
        "arnaud lindblad": "LIN",
        "lindblad": "LIN",
        "yuki tsunoda": "TSU",
        "tsunoda": "TSU",
        "valtteri bottas": "BOT",
        "bottas": "BOT",
        "isack hadjar": "HAD",
        "hadjar": "HAD",
    }

    @classmethod
    def normalize_driver_name(cls, name: str) -> str | None:
        """Normalize any driver name format to FastF1 abbreviation."""
        if not name:
            return name

        # Already an abbreviation?
        if name.upper() in cls.DRIVER_MAP:
            return name.upper()

        # Check variants
        normalized = name.lower().strip()
        if normalized in cls.NAME_VARIANTS:
            return cls.NAME_VARIANTS[normalized]

        # Check reverse map (full name)
        if name in cls.REVERSE_MAP:
            return cls.REVERSE_MAP[name]

        # Try case-insensitive match on full names
        for full_name, abbr in cls.REVERSE_MAP.items():
            if full_name.lower() == normalized:
                return abbr

        # No match found - log warning
        logger.warning("Could not normalize driver name: %s", name)
        return name

    @classmethod
    def get_full_name(cls, abbreviation: str) -> str:
        """Get full name from abbreviation."""
        return cls.DRIVER_MAP.get(abbreviation.upper(), abbreviation)

    @classmethod
    def normalize_result_list(cls, results: list) -> list:
        """Normalize all driver names in a result list to FastF1 abbreviations."""
        normalized = []
        for result in results:
            normalized_result = result.copy()
            if "driver" in result:
                normalized_result["driver"] = cls.normalize_driver_name(result["driver"])
            normalized.append(normalized_result)
        return normalized

    @classmethod
    def add_driver(cls, abbreviation: str, full_name: str) -> None:
        """Add a new driver mapping for mid-season changes or reserves."""
        cls.DRIVER_MAP[abbreviation.upper()] = full_name
        cls.REVERSE_MAP[full_name] = abbreviation.upper()
        cls.NAME_VARIANTS[full_name.lower()] = abbreviation.upper()
        logger.info("Added driver mapping: %s -> %s", abbreviation, full_name)

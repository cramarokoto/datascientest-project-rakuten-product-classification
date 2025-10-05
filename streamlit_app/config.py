"""

Config file for Streamlit App

"""

from member import Member
from pathlib import Path

TITLE = "Classification de produits Rakuten"

TEAM_MEMBERS = [
    Member(
        name="Christelle RAMAROKOTO",
        linkedin_url="https://www.linkedin.com/in/ecramarokoto/",
        github_url="https://github.com/cramarokoto",
    ),
    Member(
        name = "Cansu YILDIRIM-BALATAN",
        linkedin_url = "https://www.linkedin.com/in/cansu-balatan"
    ),
    Member(
        name="Maël ZAMORA",
        linkedin_url="https://www.linkedin.com/in/mael-zamora/"
    ),
]

PROMOTION = "Promotion Data scientist - Juillet 2025"

# Chemin de base de l'application
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"  # Si vous avez des données

def get_asset_path(filename):
    """Retourne le chemin complet d'un asset"""
    return str(ASSETS_DIR / filename)

def get_data_path(filename):
    """Retourne le chemin complet d'un fichier de données"""
    return str(DATA_DIR / filename)
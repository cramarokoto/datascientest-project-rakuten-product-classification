"""

Config file for Streamlit App

"""

from member import Member


TITLE = "My Awesome App"

TEAM_MEMBERS = [
    Member(
        name="Christelle RAMAROKOTO",
        linkedin_url="https://www.linkedin.com/in/ecramarokoto/",
        github_url="https://github.com/cramarokoto",
    ),
    Member("Cansu YILDIRIM-BALATAN"),
    Member(
        name="Maël ZAMORA",
        linkedin_url="https://www.linkedin.com/in/mael-zamora/"
    ),
]

PROMOTION = "Promotion Bootcamp Machine Learning Engineer - Juin 2025"

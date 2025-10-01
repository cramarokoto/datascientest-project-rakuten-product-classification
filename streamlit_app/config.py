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

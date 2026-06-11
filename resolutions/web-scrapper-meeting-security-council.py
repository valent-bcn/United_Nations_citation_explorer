import csv
import requests
from bs4 import BeautifulSoup

# ----------------------------
# SETTINGS
# ----------------------------
URL = "https://research.un.org/en/docs/sc/quick/meetings/" # + Year
YEARS = [str(y) for y in range(1946, 2027)]
OUTPUT_CSV = "./sc_resolutions_1946_2026.csv"
REQUEST_TIMEOUT = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ----------------------------
# HELPERS
# ----------------------------
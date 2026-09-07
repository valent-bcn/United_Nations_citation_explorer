"""
Download OHCHR Human Rights Council resolution/decision PDFs based on a
dataframe of text numbers.

For a "Text number" like '65/20', the script visits:
    https://docs.un.org/en/A/HRC/RES/65/20
and saves the resulting PDF locally as:
    65-20.pdf
"""

import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV = '../../resolutions/ohchr_resolutions.csv'
DOWNLOAD_DIR = os.path.abspath("./ohchr_pdfs")
BASE_URL = "https://docs.un.org/en/A/HRC/RES/"

# Optional path to an existing Firefox profile directory to use as a base
# (e.g. one that's already logged in / has cookies you need). If unset, a
# fresh temporary profile is created for each run.
PROFILE_PATH = os.getenv("FIREFOX_PROFILE_PATH")

MIN_WAIT_SECONDS = 4      # polite delay between requests
MAX_WAIT_SECONDS = 8
DOWNLOAD_TIMEOUT = 60      # max seconds to wait for a single PDF to finish

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# FILTERING (as specified)
# ---------------------------------------------------------------------------
def apply_filters(ohchr_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out General Assembly sessions, Presidential Statements, and Decisions."""
    df = ohchr_df.copy()
    # Session number with GA means general assembly: filter out
    df = df[~df['Session number'].astype(str).str.contains('GA', na=False)]

    # Remove Presidential Statements
    df = df[~df['Text type'].str.lower().str.contains('presidential', na=False)]

    # Remove Decisions
    df = df[~df['Text type'].str.lower().str.contains('decision', na=False)]

    # Remove report like resolutions e.g., A/C.3/74/L.31/Rev.1
    df = df[~df['Text number'].str.lower().str.contains('.', regex=False, na=False)]

    return df


# ---------------------------------------------------------------------------
# SELENIUM SETUP
# ---------------------------------------------------------------------------
def make_driver(download_dir: str) -> webdriver.Firefox:
    options = FirefoxOptions()

    # FirefoxProfile(None) creates a fresh temp profile; FirefoxProfile(path)
    # uses the given directory as a base. Either way we then layer the
    # download preferences on top.
    profile = FirefoxProfile(PROFILE_PATH)

    profile.set_preference("browser.download.folderList", 2)  # 2 = use custom dir below
    profile.set_preference("browser.download.dir", download_dir)
    profile.set_preference("browser.download.useDownloadDir", True)
    profile.set_preference("browser.download.viewableInternally.enabledTypes", "")

    # Disable Firefox's built-in PDF.js viewer so PDFs aren't rendered
    # inline - this is the Firefox equivalent of the Chrome
    # "always_open_pdf_externally" pref.
    profile.set_preference("pdfjs.disabled", True)

    # Auto-save these MIME types instead of showing the "what should Firefox
    # do with this file?" dialog.
    profile.set_preference(
        "browser.helperApps.neverAsk.saveToDisk",
        "application/pdf,application/x-pdf,application/octet-stream",
    )
    profile.set_preference("browser.helperApps.alwaysAsk.force", False)

    options.profile = profile

    driver = webdriver.Firefox(options=options)
    return driver

# ---------------------------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------------------------
def snapshot_dir(download_dir: str) -> set:
    return set(os.listdir(download_dir))


def wait_for_new_download(download_dir: str, files_before: set, timeout: int = DOWNLOAD_TIMEOUT):
    """
    Poll the download directory until a new, fully-downloaded file appears
    (i.e. not a Firefox .part partial file), then return its name.
    Returns None if the timeout is reached.
    """
    elapsed = 0
    interval = 1
    while elapsed < timeout:
        files_now = set(os.listdir(download_dir))
        new_files = files_now - files_before

        in_progress = [f for f in new_files if f.endswith(".part")]
        completed = [f for f in new_files if not f.endswith(".part")]

        if completed and not in_progress:
            return completed[0]

        time.sleep(interval)
        elapsed += interval

    return None


def download_text(driver: webdriver.Firefox, text_number: str, download_dir: str) -> bool:
    target_filename = text_number.replace("/", "-") + ".pdf"
    target_path = os.path.join(download_dir, target_filename)

    if os.path.exists(target_path):
        print(f"[SKIP] {target_filename} already exists.")
        return True

    url = BASE_URL + text_number
    print(f"[GET]  {text_number} -> {url}")

    files_before = snapshot_dir(download_dir)

    try:
        driver.get(url)
    except Exception as e:
        print(f"[ERROR] Could not load {url}: {e}")
        return False

    downloaded_name = wait_for_new_download(download_dir, files_before)

    if downloaded_name is None:
        print(f"[FAIL] Timed out waiting for download of {text_number}")
        return False

    downloaded_path = os.path.join(download_dir, downloaded_name)

    # Rename to the requested convention (e.g. 65-20.pdf), regardless of
    # whatever filename the server/Firefox originally used.
    try:
        os.replace(downloaded_path, target_path)
        print(f"[OK]   Saved as {target_filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not rename {downloaded_name} -> {target_filename}: {e}")
        return False


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(ohchr_df: pd.DataFrame):
    filtered_df = apply_filters(ohchr_df)
    text_numbers = filtered_df['Text number'].dropna().astype(str).unique().tolist()

    print(f"{len(text_numbers)} unique text numbers to download "
          f"(after filtering out GA sessions, presidential statements, and decisions).")

    driver = make_driver(DOWNLOAD_DIR)

    try:
        for i, text_number in enumerate(text_numbers, start=1):
            print(f"\n--- [{i}/{len(text_numbers)}] {text_number} ---")
            download_text(driver, text_number, DOWNLOAD_DIR)

            # Polite randomized wait before the next request (skip after the last one)
            if i < len(text_numbers):
                wait_time = random.uniform(MIN_WAIT_SECONDS, MAX_WAIT_SECONDS)
                print(f"Waiting {wait_time:.1f}s before next request...")
                time.sleep(wait_time)
    finally:
        driver.quit()


if __name__ == "__main__":

    ohchr_df = pd.read_csv(INPUT_CSV)
    run(ohchr_df) # Collect the pdfs and save them
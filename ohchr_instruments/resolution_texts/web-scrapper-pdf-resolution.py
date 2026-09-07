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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_CSV = '../../resolutions/ohchr_resolutions.csv'
DOWNLOAD_DIR = os.path.abspath("./ohchr_pdfs")
BASE_URL = "https://docs.un.org/en/A/HRC/RES/"

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

    # just for testing
    #TODO: remove this line when confirming it works
    df = df.head(10)

    return df


# ---------------------------------------------------------------------------
# SELENIUM SETUP
# ---------------------------------------------------------------------------
def make_driver(download_dir: str) -> webdriver.Chrome:
    options = Options()

    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        # This was the bug: False tells Chrome to open PDFs in its built-in
        # viewer (which is what you were seeing render on screen). True
        # forces Chrome to treat PDF responses as downloads instead.
        "plugins.always_open_pdf_externally": True,
        #"profile.default_content_setting_values.automatic_downloads": 1,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    # Belt-and-suspenders: explicitly set download behavior via CDP too.
    # This is the more reliable mechanism (works in headless mode as well,
    # in case you switch to that later), independent of the prefs above.
    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
        "behavior": "allow",
        "downloadPath": download_dir,
    })

    return driver

# ---------------------------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------------------------
def snapshot_dir(download_dir: str) -> set:
    return set(os.listdir(download_dir))


def wait_for_new_download(download_dir: str, files_before: set, timeout: int = DOWNLOAD_TIMEOUT):
    """
    Poll the download directory until a new, fully-downloaded file appears
    (i.e. not a Chrome .crdownload/.tmp partial file), then return its name.
    Returns None if the timeout is reached.
    """
    elapsed = 0
    interval = 1
    while elapsed < timeout:
        files_now = set(os.listdir(download_dir))
        new_files = files_now - files_before

        in_progress = [f for f in new_files if f.endswith(".crdownload") or f.endswith(".tmp")]
        completed = [f for f in new_files if not f.endswith(".crdownload") and not f.endswith(".tmp")]

        if completed and not in_progress:
            return completed[0]

        time.sleep(interval)
        elapsed += interval

    return None

def download_text(driver: webdriver.Chrome, text_number: str, download_dir: str) -> bool:

    target_filename = text_number.replace("/", "-") + ".pdf"
    target_path = os.path.join(download_dir, target_filename)

    if os.path.exists(target_path):
        print(f"[SKIP] {target_filename} already exists.")
        return True

    url = BASE_URL + text_number
    print(f"[GET]  {text_number} -> {url}")

    files_before = snapshot_dir(download_dir)

    # ---------------------------------------------------------
    # 1. Load UN document viewer
    # ---------------------------------------------------------
    try:
        driver.get(url)
        print(f"[LOADED] {driver.current_url}")
    except Exception as e:
        print(f"[ERROR] driver.get() failed: {e}")
        return False

    # ---------------------------------------------------------
    # 2. Find the PDF URL
    # ---------------------------------------------------------
    try:
        # Look for any link containing ".pdf"
        links = driver.find_elements(By.CSS_SELECTOR, "a[href]")

        pdf_url = None

        for link in links:
            href = link.get_attribute("href")

            print(href) #TODO: none is a pdf link, the 6 links are basecally the current url

            if href and ".pdf" in href.lower():
                pdf_url = href
                break

        if pdf_url is None:
            print("[ERROR] No PDF link found on Document Viewer.")

            # Useful diagnostic
            print("[DEBUG] Number of links:", len(links))

            return False

        print(f"[PDF]   {pdf_url}")

    except Exception as e:
        print(f"[ERROR] Could not extract PDF URL: {e}")
        return False

    # ---------------------------------------------------------
    # 3. Navigate directly to PDF
    # ---------------------------------------------------------
    try:
        driver.get(pdf_url)
        print("[OPEN]  PDF")
    except Exception as e:
        print(f"[ERROR] Could not open PDF: {e}")
        return False

    # ---------------------------------------------------------
    # 4. Wait for download
    # ---------------------------------------------------------
    downloaded_name = wait_for_new_download(
        download_dir,
        files_before
    )

    if downloaded_name is None:
        print(f"[FAIL] Timed out waiting for download of {text_number}")
        return False

    downloaded_path = os.path.join(download_dir, downloaded_name)

    # ---------------------------------------------------------
    # 5. Rename
    # ---------------------------------------------------------
    try:
        os.replace(downloaded_path, target_path)
        print(f"[OK]   Saved as {target_filename}")
        return True

    except Exception as e:
        print(
            f"[ERROR] Could not rename "
            f"{downloaded_name} -> {target_filename}: {e}"
        )
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
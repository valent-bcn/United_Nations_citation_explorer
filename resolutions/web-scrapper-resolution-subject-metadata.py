import os
from dotenv import load_dotenv
import pandas as pd
import time
import random
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------------------
# SETTINGS
# ----------------------------
INPUT_CSV = "./ga_resolutions_1946_2019_OLD.csv"
OUTPUT_CSV = "./ga_resolutions_1946_2019-subjects.csv"
load_dotenv()
BATCH_SIZE = 5000
PROFILE_PATH = os.getenv("FIREFOX_PROFILE_PATH")
BASE_DOMAIN = "https://documents.un.org/"

# Example for resolution 45/111: https://documents.un.org/symbol-explorer?s=A/RES/45/111
# How long (seconds) to wait for you to click the "I am human" button
# before giving up on a page. 
HUMAN_CLICK_TIMEOUT = 60

# ----------------------------
# LOAD DATA
# ----------------------------
df_full = pd.read_csv(INPUT_CSV)
df_full = df_full[["res_id2_unlet", "session_reg"]].drop_duplicates().reset_index(drop=True)
print(f"Total unique resolutions: {len(df_full)}")

# OUTPUT_CSV may contain already scrapped resolution, we may want to skip those
if os.path.exists(OUTPUT_CSV):
    df_done = pd.read_csv(OUTPUT_CSV)
    done_ids = set(df_done["res_id"].dropna().unique())
    print(f"Already scraped: {len(done_ids)} — skipping these.")
else:
    done_ids = set()

df_todo = df_full[~df_full["res_id2_unlet"].isin(done_ids)].reset_index(drop=True)
print(f"Remaining to scrape: {len(df_todo)}")

# We filter by the resolution we still need to scrape
df_batch = df_todo.iloc[:BATCH_SIZE].reset_index(drop=True)
print(f"This run: {len(df_batch)} resolutions (batch size={BATCH_SIZE})")


# ----------------------------
# BROWSER SETUP
# ----------------------------
profile = FirefoxProfile(PROFILE_PATH)
profile.set_preference("dom.webdriver.enabled", False)
profile.set_preference("useAutomationExtension", False)
profile.set_preference("general.useragent.override",
                       "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0")

options = Options()
options.profile = profile

# We set fixed small width and height so it doesn't bother workflow while running the code
options.add_argument("--width=800")
options.add_argument("--height=600")

results = []

# ----------------------------
# SOUND ALERT
# ----------------------------
def play_warning():
    """Gentle alert sound — cross-platform.
    Problably is going to raise an Error when not run in a linux machine.
    """
    try:
        # Linux (freedesktop systems — Ubuntu, Fedora, etc.)
        os.system("paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga 2>/dev/null &")
        time.sleep(1)
        os.system("paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga 2>/dev/null &")
        time.sleep(1)
        os.system("paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga 2>/dev/null &")
    except Exception:
        print("\a")  # fallback: terminal bell

# ----------------------------
# HELPERS
# ----------------------------

def patch_webdriver_flag(driver):
    """ For better chances not to be detected as a bot, better we remove Selenium s navigator.webdriver fingerprint."""
    try:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except Exception:
        pass

def is_security_page(driver):
    """ We rely on general patterns in website text. """
    try:
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        return (
                "security verification" in body
                or "verifies you are not a bot" in body
                or "performing security" in body
                or "i am human" in body  # catches the button label itself
                or "checking your browser" in body
        )
    except Exception:
        return False


def wait_for_real_page(driver, timeout=HUMAN_CLICK_TIMEOUT):
    start = time.time()
    alerted = False

    while time.time() - start < timeout:
        if not is_security_page(driver):
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
                if len(body_text) > 100:
                    return True
            except Exception:
                pass

        # Human verification detected
        if not alerted and is_security_page(driver):
            play_warning()

            print("\n  *** HUMAN VERIFICATION NEEDED ***")
            print("  >>> Please click the 'I am human' button in the browser window.")
            print(f"  >>> You have {timeout}s. The script will continue automatically once done.")
            alerted = True

        time.sleep(2)

    return False

# ----------------------------
# SCRAPE FIELDS
# ----------------------------
def scrape_table_field(driver, label):
    """Find a table row by its first-cell label and return the text of the data cell(s)."""
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells and cells[0].text.strip().lower() == label.lower():
                # Return all text from remaining cells joined (handles multi-cell rows)
                return " ".join(c.text.strip() for c in cells[1:] if c.text.strip()) or None
    except Exception:
        return None
    return None


def scrape_downloads(driver):
    """Sum all numeric download counts across language columns."""
    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells and "downloads" in cells[0].text.strip().lower():
                total = 0
                for cell in cells[1:]:
                    txt = cell.text.strip()
                    if txt.isdigit():
                        total += int(txt)
                # Better avoid forcing None when no downloads have taken place. We may want to sort it and plot it in the future.
                return total
    except Exception:
        return None
    return None


# ----------------------------
# SCRAPE LOOP  - BATCH SIZED
# ----------------------------
for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch)):

    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    # Minimize after launch — undetectable by server, purely OS-level
    driver.minimize_window()

    wait = WebDriverWait(driver, 30)
    url = "https://documents.un.org/symbol-explorer?s=A/RES/" + row["res_id2_unlet"].replace(" ", "")
    print(f"\n[{idx}] Processing: {row['res_id2_unlet']}")

    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        patch_webdriver_flag(driver)

        real_page_loaded = wait_for_real_page(driver, timeout=HUMAN_CLICK_TIMEOUT)

        if not real_page_loaded:
            print(f"  [skip] Timed out waiting for real page: {url}")
            results.append({
                "res_id": row["res_id2_unlet"], "title": None, "symbol": None,
                "session_year": None, "subject": None, "downloads": 0, "url": url,
            })
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, mode='a',
                                         header=not os.path.exists(OUTPUT_CSV))
            results = []  # flush
            continue

        time.sleep(random.uniform(3.0, 5.0))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")

        title = scrape_table_field(driver, "Title")
        symbol = scrape_table_field(driver, "Symbol")
        session = scrape_table_field(driver, "Session / Year")
        subject = scrape_table_field(driver, "Subject(s)")
        downloads = scrape_downloads(driver)

        results.append({
            "res_id": row["res_id2_unlet"], "title": title, "symbol": symbol,
            "session_year": session, "subject": subject, "downloads": downloads, "url": url,
        })

        # Append this row immediately to OUTPUT_CSV (checkpoint-safe)
        pd.DataFrame(results[-1:]).to_csv(
            OUTPUT_CSV, index=False,
            mode='a',  # append, never overwrite
            header=not os.path.exists(OUTPUT_CSV)  # write header only if file is new
        )

        inter_delay = random.uniform(1.0, 2.0)
        print(f"  OK — waiting {inter_delay:.1f}s before next url...")
        time.sleep(inter_delay)

    except Exception as e:
        print(f"  Error on {url}: {e}")
        row_data = {
            "res_id": row["res_id2_unlet"], "title": None, "symbol": None,
            "session_year": None, "subject": None, "downloads": 0, "url": url,
        }
        pd.DataFrame([row_data]).to_csv(
            OUTPUT_CSV, index=False,
            mode='a', header=not os.path.exists(OUTPUT_CSV)
        )

    finally:
        driver.quit()

# Final save
print(f"\nBatch done. {len(df_batch)} processed. Rerun to scrap other resolutions.")

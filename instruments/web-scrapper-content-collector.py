import os
from dotenv import load_dotenv
import pandas as pd
from urllib.parse import urljoin
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
INPUT_CSV = "./ohchr_instruments.csv"
OUTPUT_CSV = "./ohchr_instruments_detailed.csv"
PDF_DIR = ""
load_dotenv()
PROFILE_PATH = os.getenv("FIREFOX_PROFILE_PATH")
BASE_DOMAIN = "https://www.ohchr.org"

# How long (seconds) to wait for you to click the "I am human" button
# before giving up on a page. 60s is generous — lower it if you want.
HUMAN_CLICK_TIMEOUT = 60

os.makedirs(PDF_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(INPUT_CSV)

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
results = []

# ----------------------------
# HELPERS
# ----------------------------

def patch_webdriver_flag(driver):
    """Remove Selenium's navigator.webdriver fingerprint. Must be called after every get()."""
    try:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except Exception:
        pass


def is_security_page(driver):
    """Return True if the current page is still the security interstitial."""
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
    """
    Wait for the security interstitial to clear.

    - If the page clears on its own (automatic JS verification): great.
    - If a 'I am human' button appeared: print a prompt so you can click it,
      then keep polling until the real page loads or timeout expires.

    Returns True if the real page loaded, False if timed out.
    """
    start = time.time()
    alerted = False

    while time.time() - start < timeout:
        if not is_security_page(driver):
            # Extra safety: make sure there's actual content, not just a blank page
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
                if len(body_text) > 100:
                    return True
            except Exception:
                pass

        if not alerted and is_security_page(driver):
            print("\n  *** HUMAN VERIFICATION NEEDED ***")
            print("  >>> Please click the 'I am human' button in the browser window.")
            print(f"  >>> You have {timeout}s. The script will continue automatically once done.")
            alerted = True

        time.sleep(2)

    return False


def safe_find_text(driver, css_selector):
    try:
        return driver.find_element(By.CSS_SELECTOR, css_selector).text.strip() or None
    except Exception:
        return None


def safe_find_attr(driver, css_selector, attr):
    try:
        return driver.find_element(By.CSS_SELECTOR, css_selector).get_attribute(attr)
    except Exception:
        return None


# ----------------------------
# SCRAPE LOOP
# ----------------------------
for idx, row in tqdm(df.iterrows(), total=len(df)):

    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )

    wait = WebDriverWait(driver, 30)

    url = row["url"]
    title = row["title"]

    print(f"\n[{idx}] Processing: {title}")

    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        patch_webdriver_flag(driver)  # Re-patch after every navigation

        # Wait for real page — pauses for human click if needed
        real_page_loaded = wait_for_real_page(driver, timeout=HUMAN_CLICK_TIMEOUT)

        if not real_page_loaded:
            print(f"  [skip] Timed out waiting for real page: {title}")
            results.append({
                "title": title, "url": url,
                "adopted_by": "SECURITY_BLOCK", "content": None, "pdf_url": None,
            })
            continue

        # Short human-like pause after page loads
        time.sleep(random.uniform(2.5, 4.5))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
        time.sleep(random.uniform(1.0, 2.0))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")

        # ----------------------------
        # SCRAPE FIELDS
        # ----------------------------
        adopted_by = safe_find_text(driver, "div.content-hero p")
        content = safe_find_text(driver, "div.main-content")

        pdf_url = safe_find_attr(driver, "div.paragraph--type--file-version a.pdf", "href")
        if pdf_url:
            if not pdf_url.startswith("http"):
                pdf_url = urljoin(BASE_DOMAIN, pdf_url)
            print(f"  PDF: {pdf_url}")
        else:
            print("  No PDF found")

        results.append({
            "title": title,
            "url": url,
            "adopted_by": adopted_by,
            "content": content,
            "pdf_url": pdf_url,
        })

        # Save incrementally so a crash doesn't lose all progress
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

        inter_delay = random.uniform(8.0, 14.0)
        print(f"  OK — waiting {inter_delay:.1f}s before next page...")
        time.sleep(inter_delay)

    except Exception as e:
        print(f"  Error on {title}: {e}")
        results.append({
            "title": title, "url": url,
            "adopted_by": "ERROR", "content": str(e), "pdf_url": None,
        })

    finally:
        driver.quit()
        time.sleep(3)

# Final save
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print("\nDone. Saved to:", OUTPUT_CSV)
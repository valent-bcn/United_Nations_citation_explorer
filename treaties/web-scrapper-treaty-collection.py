"""
UNTC (UN Treaty Collection) title-search scraper.

For each string in SEARCH_TERMS, this script:
  1. Loads the UNTSOnline title-search page.
  2. Types the term into the "Title/Keyword" field and submits the search.
  3. Scrapes the resulting table(s) (following pagination if present).
  4. Tags every scraped row with a `search_term` column equal to that term.
  5. Appends the term's rows to a master DataFrame.

At the end, the master DataFrame (all terms stacked) is saved to CSV/XLSX.

Requirements:
    pip install selenium webdriver-manager pandas lxml beautifulsoup4

You also need a Chrome/Chromium browser installed locally. This script
was NOT executed in this sandbox (no browser/network access to the site
from here) -- run it on your own machine.
"""

import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

URL = "https://treaties.un.org/Pages/UNTSOnline.aspx?id=2&clang=_en"

# List of search terms -- each becomes both the search query and the
# identifier tagged onto its resulting rows.
SEARCH_TERMS = [
    "Convention on Biological Diversity",
    "Paris Agreement",
    # add more titles/keywords here
]

TITLE_INPUT_ID = "ctl00_ctl00_ContentPlaceHolder1_ContentPlaceHolderInnerPage_txtTitle"

# Results table id, inferred from the __doPostBack('...dgSearch', ...) calls
# in the sort-column links. If UNTC changes this id, the fallback logic
# below (find-by-header-text) will still find the table.
RESULTS_TABLE_ID = "ctl00_ctl00_ContentPlaceHolder1_ContentPlaceHolderInnerPage_dgSearch"

# If pressing Enter in the title field doesn't trigger the search on your
# render of the page, set this to the actual search button's id (inspect
# the page and look for something like "...btnSearch" or similar) and the
# script will click it instead.
SEARCH_BUTTON_ID = None  # e.g. "ctl00_ctl00_ContentPlaceHolder1_ContentPlaceHolderInnerPage_btnSearch"

PAGE_LOAD_TIMEOUT = 20
HEADLESS = False  # set True once you've confirmed it works, for speed

OUTPUT_CSV = "untc_title_search_results.csv"


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def make_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def find_results_table_html(driver):
    """Return the outer HTML of the results table, or None if not found."""
    # First try the known id.
    try:
        table = driver.find_element(By.ID, RESULTS_TABLE_ID)
        return table.get_attribute("outerHTML")
    except NoSuchElementException:
        pass

    # Fallback: search all tables for one whose header row contains
    # "Registration" and "Title" (matches the header row you showed).
    soup = BeautifulSoup(driver.page_source, "lxml")
    for table in soup.find_all("table"):
        header_text = table.get_text(" ", strip=True)
        if "Registration" in header_text and "Title" in header_text and "Conclusion" in header_text:
            return str(table)

    return None


def parse_table(table_html, search_term):
    """Parse the results table HTML into a DataFrame, tagged with search_term."""
    if table_html is None:
        return pd.DataFrame()

    soup = BeautifulSoup(table_html, "lxml")

    # Grab visible header labels from the first row (<th> cells).
    header_row = soup.find("tr")
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")] if header_row else []

    rows = []
    for tr in soup.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if not cells:
            continue
        row = {}
        for i, td in enumerate(cells):
            col_name = headers[i] if i < len(headers) and headers[i] else f"col_{i}"
            # "Participants" column mostly just holds a "See Details" link;
            # capture both the visible text and the href if present.
            link = td.find("a")
            if link and link.get("href"):
                row[col_name] = link.get_text(strip=True) or "See Details"
                row[f"{col_name}_link"] = link["href"]
            else:
                row[col_name] = td.get_text(strip=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "search_term", search_term)
    return df


def go_to_next_page(driver, current_page_num):
    """
    UNTC results pagers are typically rendered as numbered postback links
    inside the results table (e.g. '2', '3', ...) or '...' for jumps.
    Returns True if it navigated to a new page, False if there isn't one.
    """
    next_num = str(current_page_num + 1)
    try:
        link = driver.find_element(
            By.XPATH,
            f"//a[contains(@href, \"__doPostBack\") and normalize-space(text())='{next_num}']",
        )
    except NoSuchElementException:
        return False

    old_table = find_results_table_html(driver)
    link.click()

    try:
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            lambda d: find_results_table_html(d) != old_table
        )
    except TimeoutException:
        return False

    return True


def search_title(driver, term):
    """Load the search page fresh and submit a title search for `term`."""
    driver.get(URL)

    title_input = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        EC.presence_of_element_located((By.ID, TITLE_INPUT_ID))
    )
    title_input.clear()
    title_input.send_keys(term)

    if SEARCH_BUTTON_ID:
        driver.find_element(By.ID, SEARCH_BUTTON_ID).click()
    else:
        title_input.send_keys(Keys.RETURN)

    # Wait for either the results table or a "no results" state to appear.
    WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
        lambda d: find_results_table_html(d) is not None
        or "no records" in d.page_source.lower()
    )


def scrape_all_pages_for_term(driver, term):
    dfs = []
    page_num = 1
    while True:
        table_html = find_results_table_html(driver)
        df = parse_table(table_html, term)
        if not df.empty:
            dfs.append(df)

        try:
            moved = go_to_next_page(driver, page_num)
        except StaleElementReferenceException:
            moved = False

        if not moved:
            break
        page_num += 1
        time.sleep(0.5)  # be polite / let the DOM settle

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    driver = make_driver()
    all_results = []

    try:
        for term in SEARCH_TERMS:
            print(f"Searching: {term!r} ...")
            try:
                search_title(driver, term)
                term_df = scrape_all_pages_for_term(driver, term)
                print(f"  -> {len(term_df)} rows")
                all_results.append(term_df)
            except TimeoutException:
                print(f"  !! Timed out waiting for results for {term!r}, skipping.")
            except Exception as e:
                print(f"  !! Error on {term!r}: {e}")
    finally:
        driver.quit()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(final_df)} total rows to {OUTPUT_CSV}")
    return final_df


if __name__ == "__main__":
    main()
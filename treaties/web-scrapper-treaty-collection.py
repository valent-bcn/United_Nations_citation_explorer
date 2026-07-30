"""
UNTC (UN Treaty Collection) title-search scraper.

OLD version, kept for testing.
In order to use the up to date version, use the Class UNTCSearcher
"""

import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
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

BASE_URL = "https://treaties.un.org"
URL = f"{BASE_URL}/Pages/UNTSOnline.aspx?id=2&clang=_en"

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

# Delay (seconds) between requests to each treaty's details page, to be
# polite to the server. Increase if you hit rate limiting.
DETAILS_FETCH_DELAY = 0.5


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
            link = td.find("a")
            if col_name == "Participants" and link and link.get("href"):
                # This column is just a "See Details" link to the treaty's
                # detail page -- store the absolute URL as its own column
                # instead of the (uninformative) "See Details" text.
                row["url"] = urljoin(BASE_URL, link["href"])
            elif link and link.get("href"):
                row[col_name] = link.get_text(strip=True)
                row[f"{col_name}_link"] = urljoin(BASE_URL, link["href"])
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


def _norm(text):
    """Normalize header/label text for loose matching (nbsp, case, spacing)."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def fetch_details(session, url):
    """
    Fetch a treaty's showDetails.aspx page and extract:
        - full_title
        - place
        - date
        - text_document_urls (list)
        - volume_pdf_urls (list)
    Returns a dict; missing fields are left as None / empty list.
    """
    result = {
        "full_title": None,
        "place": "",
        "date": "",
        "text_document_urls": [],
        "volume_pdf_urls": [],
    }
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    !! Failed to fetch details page {url}: {e}")
        return result

    soup = BeautifulSoup(resp.text, "lxml")

    # --- Full title ---
    title_span = soup.find(id="lblTitle1")
    if title_span:
        result["full_title"] = title_span.get_text(strip=True)

    # --- Place / Date table (the "Places/dates of conclusion" table, id="dgsign") ---
    sign_table = soup.find("table", id="dgsign")
    if sign_table is None:
        # Fallback: any table whose header row is exactly Place / Date.
        for table in soup.find_all("table"):
            header_texts = [_norm(th.get_text()) for th in table.find_all("th")]
            if "place" in header_texts and "date" in header_texts:
                sign_table = table
                break

    if sign_table is not None:
        places, dates = [], []
        data_rows = sign_table.find_all("tr")[1:]  # skip header row
        for row in data_rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                place_text = cells[0].get_text(strip=True)
                date_text = cells[1].get_text(strip=True)
                if place_text:
                    places.append(place_text)
                if date_text:
                    dates.append(date_text)
        result["place"] = "; ".join(places)
        result["date"] = "; ".join(dates)

    # --- Generic label -> value rows (th/td pairs), used for the PDF links ---
    for th in soup.find_all("th"):
        label = _norm(th.get_text())
        if not label:
            continue
        td = th.find_next_sibling("td")
        if td is None:
            continue

        hrefs = [urljoin(BASE_URL, a["href"]) for a in td.find_all("a", href=True) if a.get("href")]

        if "text document" in label:
            result["text_document_urls"].extend(hrefs)
        elif "volume" in label and "pdf" in label:
            result["volume_pdf_urls"].extend(hrefs)

    return result


def enrich_with_details(df, delay=0.5):
    """
    For each unique 'url' in df, fetch the treaty's details page and merge
    in full_title / place / date / text_document_urls / volume_pdf_urls.
    """
    if df.empty or "url" not in df.columns:
        return df

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; UNTC-scraper/1.0)"})

    unique_urls = df["url"].dropna().unique().tolist()
    print(f"Fetching details for {len(unique_urls)} unique treaty pages...")

    details_by_url = {}
    for i, url in enumerate(unique_urls, 1):
        print(f"  [{i}/{len(unique_urls)}] {url}")
        details_by_url[url] = fetch_details(session, url)
        time.sleep(delay)  # be polite to the server

    details_df = (
        pd.DataFrame.from_dict(details_by_url, orient="index")
        .reset_index()
        .rename(columns={"index": "url"})
    )
    # Join list columns into readable strings for CSV/XLSX friendliness.
    details_df["text_document_urls"] = details_df["text_document_urls"].apply(lambda x: "; ".join(x) if x else "")
    details_df["volume_pdf_urls"] = details_df["volume_pdf_urls"].apply(lambda x: "; ".join(x) if x else "")

    merged = df.merge(details_df, on="url", how="left")

    # Replace the truncated search-results Title with the full title where available.
    if "Title" in merged.columns:
        merged["Title"] = merged["full_title"].where(merged["full_title"].notna(), merged["Title"])
        merged = merged.drop(columns=["full_title"])

    return merged


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

    final_df = enrich_with_details(final_df, delay=DETAILS_FETCH_DELAY)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(final_df)} total rows to {OUTPUT_CSV}")
    return final_df


if __name__ == "__main__":
    main()
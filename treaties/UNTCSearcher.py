import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager


class UNTCSearcher:

    # ---------- class attributes ----------
    BASE_URL = "https://treaties.un.org"
    URL = f"{BASE_URL}/Pages/UNTSOnline.aspx?id=2&clang=_en"

    TITLE_INPUT_ID = (
        "ctl00_ctl00_ContentPlaceHolder1_"
        "ContentPlaceHolderInnerPage_txtTitle"
    )

    RESULTS_TABLE_ID = (
        "ctl00_ctl00_ContentPlaceHolder1_"
        "ContentPlaceHolderInnerPage_dgSearch"
    )
    MESSAGE_ID = (
        "ctl00_ctl00_ContentPlaceHolder1_"
        "ContentPlaceHolderInnerPage_lblMsg"
    )

    SEARCH_BUTTON_ID = None

    PAGE_LOAD_TIMEOUT = 20
    HEADLESS = False

    DETAILS_FETCH_DELAY = 2

    def __init__(self,
                 headless=None,
                 timeout=None):

        self.headless = headless if headless is not None else self.HEADLESS
        self.timeout = timeout if timeout is not None else self.PAGE_LOAD_TIMEOUT

        self.driver = self.make_driver()

    def has_no_results(self):
        try:
            msg = self.driver.find_element(By.ID, self.MESSAGE_ID)
            return "record not found" in msg.text.lower()
        except NoSuchElementException:
            return False

    def close(self):
        self.driver.quit()

    def make_driver(self):
        opts = Options()

        if self.headless:
            opts.add_argument("--headless=new")

        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--disable-gpu")

        service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(
            service=service,
            options=opts,
        )

        driver.set_page_load_timeout(self.timeout)

        return driver

    def search(self, term):

        try:
            found = self.search_title(term)

            if not found:
                print(f"No results found for {term}.")
                return pd.DataFrame() # Empty

            df = self.scrape_all_pages_for_term(term)
            df = self.enrich_with_details(df)

            print(f" -> {len(df)} rows")

            return df

        except TimeoutException:
            print(f"Timed out waiting for results for {term}.")
            return pd.DataFrame() # Empty

        except Exception as e:
            print(f"Error searching {term}: {e}")
            return pd.DataFrame() # Empty

    def find_results_table_html(self):
        try:
            table = self.driver.find_element(By.ID, self.RESULTS_TABLE_ID)
            return table.get_attribute("outerHTML")

        except NoSuchElementException:
            pass

        soup = BeautifulSoup(self.driver.page_source, "lxml")

        for table in soup.find_all("table"):
            header = table.get_text(" ", strip=True)

            if (
                    "Registration" in header
                    and "Title" in header
                    and "Conclusion" in header
            ):
                return str(table)

        return None

    def parse_table(self, table_html, search_term):
        if table_html is None:
            return pd.DataFrame()

        soup = BeautifulSoup(table_html, "lxml")

        # Locate the real header; ignore the row of page number cells
        header_row = None
        for tr in soup.find_all("tr"):
            if tr.find("th"):
                header_row = tr
                break

        if header_row is None:
            return pd.DataFrame()

        headers = [
            th.get_text(" ", strip=True)
            for th in header_row.find_all("th")
        ]

        rows = []

        for tr in header_row.find_next_siblings("tr"):

            # Skip pagination rows
            if "pagernumber" in tr.get("class", []):
                continue

            cells = tr.find_all("td")

            if len(cells) == 0:
                continue

            # Ignore rows that obviously aren't treaty rows
            if len(cells) < len(headers):
                continue

            row = {}

            for i, td in enumerate(cells):

                col = headers[i] if i < len(headers) else f"col_{i}"

                link = td.find("a", href=True)

                if col == "Participants" and link:
                    row["url"] = urljoin(self.BASE_URL, link["href"])

                elif link:
                    row[col] = link.get_text(" ", strip=True)
                    row[f"{col}_link"] = urljoin(self.BASE_URL, link["href"])

                else:
                    row[col] = td.get_text(" ", strip=True)

            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            df.insert(0, "search_term", search_term)

        return df


    def go_to_next_page(self, current_page_num):

        next_num = str(current_page_num + 1)

        try:
            link = self.driver.find_element(
                By.XPATH,
                f"//a[contains(@href,'__doPostBack') and normalize-space(text())='{next_num}']"
            )

        except NoSuchElementException:
            return False

        old_table = self.find_results_table_html()

        link.click()

        try:
            WebDriverWait(self.driver, self.timeout).until(
                lambda d: self.find_results_table_html() != old_table
            )

        except TimeoutException:
            return False

        return True

    def search_title(self, term):

        self.driver.get(self.URL)

        title = WebDriverWait(
            self.driver,
            self.timeout,
        ).until(
            EC.presence_of_element_located((By.ID, self.TITLE_INPUT_ID))
        )

        title.clear()
        title.send_keys(term)

        if self.SEARCH_BUTTON_ID:
            self.driver.find_element(By.ID, self.SEARCH_BUTTON_ID).click()
        else:
            title.send_keys(Keys.RETURN)

        WebDriverWait(self.driver, self.timeout).until(
            lambda d: (
                self.find_results_table_html() is not None
                or self.has_no_results()
            )
        )

        if self.has_no_results():
            return False
        return True

    def scrape_all_pages_for_term(self, term):
        dfs = []
        page_num = 1

        while True:

            table_html = self.find_results_table_html()
            df = self.parse_table(table_html, term)

            if not df.empty:
                dfs.append(df)

            try:
                moved = self.go_to_next_page(page_num)

            except StaleElementReferenceException:
                moved = False

            if not moved:
                break

            page_num += 1
            time.sleep(0.5)

        if dfs:
            return pd.concat(dfs, ignore_index=True)

        return pd.DataFrame()

    @staticmethod
    def _norm(text):
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()

    def fetch_details(self, session, url):
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
                header_texts = [self._norm(th.get_text()) for th in table.find_all("th")]
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
            label = self._norm(th.get_text())
            if not label:
                continue
            td = th.find_next_sibling("td")
            if td is None:
                continue

            hrefs = [urljoin(self.BASE_URL, a["href"]) for a in td.find_all("a", href=True) if a.get("href")]

            if "text document" in label:
                result["text_document_urls"].extend(hrefs)
            elif "volume" in label and "pdf" in label:
                result["volume_pdf_urls"].extend(hrefs)

        return result

    def enrich_with_details(self, df, delay=None):
        if delay is None:
            delay = self.DETAILS_FETCH_DELAY

        if df.empty or "url" not in df.columns:
            return df

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; UNTC-scraper/1.0)"
        })

        unique_urls = df["url"].dropna().unique().tolist()

        details = {}

        for url in unique_urls:
            details[url] = self.fetch_details(session, url)
            time.sleep(delay)

        details_df = (
            pd.DataFrame.from_dict(details, orient="index")
            .reset_index()
            .rename(columns={"index": "url"})
        )

        details_df["text_document_urls"] = (
            details_df["text_document_urls"]
            .apply(lambda x: "; ".join(x) if x else "")
        )

        details_df["volume_pdf_urls"] = (
            details_df["volume_pdf_urls"]
            .apply(lambda x: "; ".join(x) if x else "")
        )

        merged = df.merge(details_df, on="url", how="left")

        if "Title" in merged.columns:
            merged["Title"] = merged["full_title"].where(
                merged["full_title"].notna(),
                merged["Title"],
            )
            merged.rename(columns={"Title": "title",
                                   "Registration Number": "registration_number",
                                   "Conclusion Date": "conclusion_date",
                                   "Entry into Force Date": "entry_into_force_date",
                                   "Treaty Type": "treaty_type"}, inplace=True)

            columns_to_drop = ["full_title"]
            if "" in merged.columns:
                columns_to_drop.append("")
            merged = merged.drop(columns=columns_to_drop)

        return merged

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
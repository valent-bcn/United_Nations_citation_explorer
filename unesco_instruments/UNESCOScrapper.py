import os
import re
import json
import time
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.parse import urljoin, urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager


class UNESCOScraper:

    BASE_URL = "https://www.unesco.org/en/legal-affairs/list"
    RESULT_LIST_CSS = "div.sdh-results a.teaser"

    # --- detail-page selectors -------------------------------------------------
    CATEGORY_CSS = "p.heading__content__type"
    PDF_DROPDOWN_CSS = "div.dropdown-menu a.dropdown-item"
    CONTENT_CSS = (
        "#block-unesco-content > article > div > "
        "div.dataset-template-wrapper > div > div.row.main-content > "
        "div.col-lg-9.content-wrapper > div:nth-child(2)"
    )
    PDF_VIEWER_IFRAME_CSS = "iframe[src*='documentViewer.xhtml']"

    DATALAYER_RE = re.compile(r"window\.dataLayer\.push\((\{.*?\})\);", re.DOTALL)

    PDF_DIR = "./pdf_files"
    TXT_DIR = "./txt"

    # To avoid redundancy in the urls, we clean the ?hub=number from urls to standard the format
    STRIP_PARAMS = {"hub"}

    def __init__(self, headless=True, timeout=30, request_delay=1.0):
        self.headless = headless
        self.timeout = timeout
        self.request_delay = request_delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

        os.makedirs(self.PDF_DIR, exist_ok=True)
        os.makedirs(self.TXT_DIR, exist_ok=True)

    @classmethod
    def clean_url(cls, url):
        """Remove internal tracking/context query params (e.g. ?hub=66535)."""
        if not url:
            return url
        parts = urlsplit(url)
        kept = [(k, v) for k, v in parse_qsl(parts.query) if k not in cls.STRIP_PARAMS]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))


    # ------------------------------------------------------------------ #
    # Selenium: results list page
    # ------------------------------------------------------------------ #
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

    def scrape(self):

        driver = self.make_driver()

        try:
            driver.get(self.BASE_URL)

            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, self.RESULT_LIST_CSS)
                )
            )

            records = []

            items = driver.find_elements(
                By.CSS_SELECTOR, self.RESULT_LIST_CSS
            )

            for item in items:

                title = item.find_element(
                    By.CSS_SELECTOR, ".h5"
                ).text.strip()

                link = item.get_attribute("href")

                try:
                    location = item.find_element(
                        By.CSS_SELECTOR, ".address-wrapper"
                    ).text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    location = ""

                try:
                    date = item.find_element(
                        By.CSS_SELECTOR, ".date"
                    ).text.strip()
                except (NoSuchElementException, StaleElementReferenceException):
                    date = ""

                records.append({
                    "title": title,
                    "date": date,
                    "location": location,
                    "url": self.clean_url(url=link),
                })

            return pd.DataFrame(records)

        finally:
            driver.quit()

    # ------------------------------------------------------------------ #
    # requests + BeautifulSoup: detail pages
    # ------------------------------------------------------------------ #
    def _parse_datalayer(self, html):
        """Extract the JSON object pushed to window.dataLayer, if present."""
        match = self.DATALAYER_RE.search(html)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    def _find_english_pdf(self, soup):
        for a in soup.select(self.PDF_DROPDOWN_CSS):
            if a.get_text(strip=True).lower() == "english":
                return a.get("href")
        return None

    def _resolve_real_pdf_url(self, html, page_url):
        """
        Given the HTML of an intermediate preview page (documentViewer wrapper)
        and the URL of that page, extract the real downloadable PDF URL from
        the embedded viewer iframe's 'file' query parameter.
        Returns None if no matching iframe/param is found.
        """
        soup = BeautifulSoup(html, "html.parser")
        iframe = soup.select_one(self.PDF_VIEWER_IFRAME_CSS)
        if iframe is None or not iframe.get("src"):
            return None

        iframe_url = urljoin(page_url, iframe["src"])
        query = parse_qs(urlparse(iframe_url).query)
        file_param = query.get("file")
        if not file_param:
            return None

        # parse_qs already url-decodes the value, e.g.:
        # /in/rest/annotationSVC/DownloadWatermarkedAttachment/attach_import_...?_=068427engo.pdf
        return urljoin(iframe_url, file_param[0])

    def _download_pdf(self, pdf_url, dest_path):
        try:
            r = self.session.get(pdf_url, timeout=self.timeout, stream=True)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type:
                # Not a direct PDF response -> likely the intermediate preview page.
                r.close()
                page = self.session.get(pdf_url, timeout=self.timeout)
                page.raise_for_status()

                real_pdf_url = self._resolve_real_pdf_url(page.text, page.url)
                if not real_pdf_url:
                    print(f"  [WARN] no PDF found at ({pdf_url}); "
                          f"got content-type={content_type!r} and no viewer iframe")
                    return False

                r = self.session.get(real_pdf_url, timeout=self.timeout, stream=True)
                r.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            print(f"  [WARN] could not download PDF ({pdf_url}): {e}")
            return False

    def fetch_details(self, url):
        """
        Fetch a single instrument detail page and extract:
            - category    : e.g. "Recommendation", "Convention", ...
            - pdf_url     : English PDF link (None if unavailable)
            - uuid        : unique content id, from window.dataLayer
            - pdf_filename: "<uuid>.pdf" (None if no uuid found)
            - content     : plain-text body of the description section

        Side effects (only when a uuid is found):
            - downloads the English PDF to ./pdf_files/<uuid>.pdf
            - writes the extracted text to ./txt/<uuid>.txt

        Returns a dict (usable as a pandas row); missing fields are None.
        """
        details = {
            "category": None,
            "pdf_url": None,
            "uuid": None,
            "pdf_filename": None,
            "has_pdf": False,
            "content": None,
        }

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] could not fetch {url}: {e}")
            return details

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # --- uuid, via the dataLayer JSON blob ---
        data = self._parse_datalayer(html)
        details["uuid"] = data.get("content", {}).get("uuid") # Fails in 2 cases:

        # --- category ---
        category_el = soup.select_one(self.CATEGORY_CSS)
        if category_el:
            details["category"] = category_el.get_text(strip=True)

        # --- English PDF link ---
        details["pdf_url"] = self.clean_url(self._find_english_pdf(soup))

        # --- body content ---
        content_el = soup.select_one(self.CONTENT_CSS)
        content_text = content_el.get_text("\n", strip=True) if content_el else None
        details["content"] = content_text

        # --- side effects: download pdf + save txt, keyed by uuid ---
        if details["uuid"]:
            details["pdf_filename"] = f"{details['uuid']}.pdf"

            if details["pdf_url"]:
                pdf_path = os.path.join(self.PDF_DIR, details["pdf_filename"])
                details["has_pdf"] = self._download_pdf(details["pdf_url"], pdf_path)

            if content_text:
                txt_path = os.path.join(self.TXT_DIR, f"{details['uuid']}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(content_text)
        else:
            print(f"  [WARN] no uuid found for {url}; pdf/txt not saved")

        return details

    def enrich(self, df):
        """
        Calls fetch_details() for every row['url'] in df and returns a new
        dataframe with the original columns plus category/pdf_url/uuid/
        pdf_filename/content.
        """
        rows = []
        total = len(df)

        for i, url in enumerate(df["url"], start=1):
            print(f"[{i}/{total}] {url}")
            rows.append(self.fetch_details(url))
            time.sleep(self.request_delay)

        details_df = pd.DataFrame(rows)
        return pd.concat([df.reset_index(drop=True), details_df], axis=1)


if __name__ == "__main__":
    scraper = UNESCOScraper(headless=False)

    df = scraper.scrape()
    print(df)

    enriched = scraper.enrich(df)
    print(enriched)

    enriched.to_csv("UNESCO_legal_instruments_detail.csv", index=False)



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
    CONTENT_CSS = (
        "#block-unesco-content > article > div > "
        "div.dataset-template-wrapper > div > div.row.main-content > "
        "div.col-lg-9.content-wrapper > div:nth-child(2)"
    )

    DATALAYER_RE = re.compile(r"window\.dataLayer\.push\((\{.*?\})\);", re.DOTALL)


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

    def fetch_details(self, url):
        """
        Fetch a single instrument detail page and extract:
            - category : e.g. "Recommendation", "Convention", ...
            - uuid     : unique content id, from window.dataLayer
            - content  : plain-text body of the description section

        Returns a dict (usable as a pandas row); missing fields are None.
        """
        details = {
            "category": None,
            "uuid": None,
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
        details["uuid"] = data.get("content", {}).get("uuid")

        # --- category ---
        category_el = soup.select_one(self.CATEGORY_CSS)
        if category_el:
            details["category"] = category_el.get_text(strip=True)

        # --- body content ---
        content_el = soup.select_one(self.CONTENT_CSS)
        details["content"] = (
            content_el.get_text("\\n", strip=True) if content_el else None
        )

        return details


    def enrich(self, df):
        """
        Calls fetch_details() for every row['url'] in df and returns a new
        dataframe with the original columns plus category/uuid/content.
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
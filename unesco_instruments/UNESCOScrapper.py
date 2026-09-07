import os
import re
import json
import time
import random
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.parse import urljoin, urlparse, parse_qs

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    CONTENT_CSS = "#description"

    FALLBACK_CONTENT_CSS = (
        "#block-unesco-content > article > div > "
        "div.dataset-template-wrapper > div > div.row.main-content > "
        "div.col-lg-9.content-wrapper > div:nth-child(2)"
    )

    DATALAYER_RE = re.compile(r"window\.dataLayer\.push\((\{.*?\})\);", re.DOTALL)

    # To avoid redundancy in the urls, we clean the ?hub=number from urls to standard the format
    STRIP_PARAMS = {"hub"}

    def __init__(
        self,
        headless=True,
        timeout=30,
        request_delay=1.5,
        delay_jitter=1.0,
        max_retries=5,
        backoff_factor=2.0,
        detail_max_attempts=5,
        detail_retry_wait=10.0,
    ):
        self.headless = headless
        self.timeout = timeout

        # base delay between successful requests, plus a random jitter added
        # on top so the pacing isn't perfectly regular / bot-like
        self.request_delay = request_delay
        self.delay_jitter = delay_jitter

        # --- application-level retry knobs for fetch_details() ---------------
        # These sit ABOVE the transport-level Retry below. The transport
        # retries live inside a single requests.get() call and are silent
        # (you never see them logged) and bounded by that one call's
        # timeout budget. A handful of URLs still time out even after those
        # internal retries are exhausted -- usually just a slow moment on
        # the server, not a dead page -- so we retry the *whole* request a
        # few more times, with a visible wait and a fresh connection.
        self.detail_max_attempts = detail_max_attempts
        self.detail_retry_wait = detail_retry_wait

        # keep these around so _new_session() can rebuild an identical adapter
        self._transport_max_retries = max_retries
        self._transport_backoff_factor = backoff_factor

        self.session = self._build_session()

    # ------------------------------------------------------------------ #
    # requests session (built once in __init__, rebuilt on retry)
    # ------------------------------------------------------------------ #
    def _build_session(self):
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

        # --- robust retry/backoff at the transport level ---------------------
        # This handles 429 (rate limit) AND other transient failures
        # (500/502/503/504, connection resets, read timeouts) uniformly,
        # instead of hand-coding a check for one specific status code.
        #
        # - total=max_retries: how many times urllib3 will retry a request
        # - backoff_factor: delay grows as {backoff_factor} * (2 ** (retry_count - 1))
        #     e.g. with backoff_factor=2 -> 2s, 4s, 8s, 16s, 32s
        # - status_forcelist: which HTTP status codes should trigger a retry
        # - respect_retry_after_header=True: if the server sends a
        #     `Retry-After` header (very common on 429s), urllib3 will sleep
        #     for exactly that long instead of guessing
        # - allowed_methods: retry GET (default excludes some methods; we're
        #     only doing GETs here so this is safe)
        retry_strategy = Retry(
            total=self._transport_max_retries,
            backoff_factor=self._transport_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            raise_on_status=False,  # let us inspect/handle the final response ourselves
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _new_session(self):
        """
        Replace self.session with a fresh one (same headers/retry policy).

        A read timeout that survives urllib3's internal retries is often a
        sign the underlying TCP connection is in a bad state (half-open,
        stuck behind a proxy, etc.) rather than the URL itself being broken.
        Starting a clean connection before the next attempt clears that up.
        """
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._build_session()

    @classmethod
    def clean_url(cls, url):
        """Remove internal tracking/context query params (e.g. ?hub=66535)."""
        if not url:
            return url
        parts = urlsplit(url)
        kept = [(k, v) for k, v in parse_qsl(parts.query) if k not in cls.STRIP_PARAMS]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))

    def _sleep_with_jitter(self, base=None):
        """Sleep for `base` (default request_delay) seconds plus a random jitter."""
        base = self.request_delay if base is None else base
        time.sleep(base + random.uniform(0, self.delay_jitter))

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

    def _parse_detail_html(self, html):
        """Parse a fetched detail page's HTML into the category/uuid/content dict."""
        soup = BeautifulSoup(html, "html.parser")

        data = self._parse_datalayer(html)
        uuid = data.get("content", {}).get("uuid")

        category = None
        category_el = soup.select_one(self.CATEGORY_CSS)
        if category_el:
            category = category_el.get_text(strip=True)

        content_el = soup.select_one(self.CONTENT_CSS)
        if content_el:
            content = content_el.get_text("\n", strip=True)
        else:
            fallback_el = soup.select_one(self.FALLBACK_CONTENT_CSS)
            content = fallback_el.get_text("\n", strip=True) if fallback_el else None

        return {"category": category, "uuid": uuid, "content": content}

    def fetch_details(self, url, max_attempts=None, retry_wait=None):
        """
        Fetch a single instrument detail page and extract:
            - category : e.g. "Recommendation", "Convention", ...
            - uuid     : unique content id, from window.dataLayer
            - content  : plain-text body of the description section

        Returns a dict (usable as a pandas row); missing fields are None.

        The Session's transport-level Retry adapter already absorbs
        429/5xx/read-timeouts *within* a single requests.get() call. This
        method adds a second, visible layer on top: if that call still ends
        up failing (its retry budget exhausted), we wait `retry_wait`
        seconds, swap in a fresh Session (to shake off any stuck
        connection), and try the whole request again, up to `max_attempts`
        times total.
        """
        max_attempts = self.detail_max_attempts if max_attempts is None else max_attempts
        retry_wait = self.detail_retry_wait if retry_wait is None else retry_wait

        details = {"category": None, "uuid": None, "content": None}

        for attempt in range(1, max_attempts + 1):
            try:
                # give slow pages a little more room on later attempts
                attempt_timeout = self.timeout + (attempt - 1) * 10
                resp = self.session.get(url, timeout=attempt_timeout)

                if resp.status_code == 429:
                    # Transport retries were exhausted and the server is
                    # still rate-limiting us. Back off hard once more.
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else self.request_delay * 10
                    print(f"  [WARN] still rate-limited (attempt {attempt}/{max_attempts}), "
                          f"waiting {wait_s:.0f}s: {url}")
                    time.sleep(wait_s)
                    resp = self.session.get(url, timeout=attempt_timeout)

                resp.raise_for_status()

            except requests.RequestException as e:
                print(f"  [WARN] attempt {attempt}/{max_attempts} failed for {url}: {e}")
                if attempt == max_attempts:
                    print(f"  [ERROR] giving up on {url} after {max_attempts} attempts")
                    return details
                time.sleep(retry_wait + random.uniform(0, self.delay_jitter))
                self._new_session()
                continue

            # success
            return self._parse_detail_html(resp.text)

        return details  # unreachable, kept for safety

    def enrich(self, df, retry_failed_pass=True):
        """
        Calls fetch_details() for every row['url'] in df and returns a new
        dataframe with the original columns plus category/uuid/content.

        If `retry_failed_pass` is True (default), any row that still has no
        content after the main loop gets one more attempt at the very end,
        after a cooldown wait. This catches pages that were having a bad
        moment even across fetch_details()'s own internal retries -- giving
        the site a longer break before trying again tends to clear these up.
        """
        rows = []
        total = len(df)

        for i, url in enumerate(df["url"], start=1):
            print(f"[{i}/{total}] {url}")
            rows.append(self.fetch_details(url))
            self._sleep_with_jitter()

        details_df = pd.DataFrame(rows)
        result = pd.concat([df.reset_index(drop=True), details_df], axis=1)

        if retry_failed_pass:
            failed_mask = result["content"].isna()
            n_failed = int(failed_mask.sum())
            if n_failed:
                print(f"\n--- retry pass: {n_failed} row(s) with no content, "
                      f"cooling down {self.detail_retry_wait:.0f}s before retrying ---")
                time.sleep(self.detail_retry_wait)
                self._new_session()

                for idx in result.index[failed_mask]:
                    url = result.at[idx, "url"]
                    print(f"[retry] {url}")
                    fresh = self.fetch_details(url)
                    for col, val in fresh.items():
                        result.at[idx, col] = val
                    self._sleep_with_jitter()

                still_failed = int(result["content"].isna().sum())
                if still_failed:
                    print(f"  [WARN] {still_failed} row(s) still have no content after the retry pass")

        return result


if __name__ == "__main__":
    scraper = UNESCOScraper(headless=False)

    df = scraper.scrape()
    print(df)

    enriched = scraper.enrich(df)
    print(enriched)

    enriched.to_csv("UNESCO_legal_instruments_detail.csv", index=False)
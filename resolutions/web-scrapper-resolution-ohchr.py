"""
Scrapes the resolutions table from the Universal Rights Group portal.
By default it is set to scrape from the following url:
https://www.universal-rights.org/human-rights/human-rights-resolutions-portal/

Last accessed on September 2026
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

logger = logging.getLogger(__name__)


class UniversalRightsScraper:

    DEFAULT_URL = "https://www.universal-rights.org/human-rights/human-rights-resolutions-portal/"
    MIN_COLUMNS = 16

    def __init__(
            self,
            url: str = DEFAULT_URL,
            headless: bool = True,
            page_load_wait: float = 10.0,
            min_columns: int = MIN_COLUMNS,
    ):
        self.url = url
        self.headless = headless
        self.page_load_wait = page_load_wait
        self.min_columns = min_columns
        self.driver: Optional[WebDriver] = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "UniversalRightsScraper":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Driver lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch the Chrome driver if it isn't already running."""
        if self.driver is not None:
            return

        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        logger.info("Starting Chrome driver")
        self.driver = webdriver.Chrome(options=options)

    def stop(self) -> None:
        """Close the driver, if running."""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
            logger.info("Chrome driver closed")

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------
    def scrape(self) -> pd.DataFrame:
        """Load the page, parse the resolutions table, and return a DataFrame."""
        if self.driver is None:
            raise RuntimeError(
                "Driver not started. Call start() first, or use the scraper "
                "as a context manager (`with UniversalRightsScraper() as s:`)."
            )

        logger.info("Fetching %s", self.url)
        self.driver.get(self.url)
        time.sleep(self.page_load_wait)  # wait for JS table rendering

        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        logger.info("Found %d raw rows", len(rows))

        data: List[Dict[str, str]] = []
        for row in rows:
            row_data = self._parse_row(row)
            if row_data is not None:
                data.append(row_data)

        logger.info("Parsed %d valid rows", len(data))
        return pd.DataFrame(data)

    def _parse_row(self, row: WebElement) -> Optional[Dict[str, str]]:
        """Parse a single <tr> element into a dict, or None if malformed."""
        cols = row.find_elements(By.TAG_NAME, "td")

        if len(cols) < self.min_columns:
            return None

        try:
            title_element = cols[5].find_element(By.TAG_NAME, "a")
            title = title_element.text.strip()
            link = title_element.get_attribute("href")

            return {
                "Year": cols[0].text.strip(),
                "Month": cols[1].text.strip(),
                "Session number": cols[2].text.strip(),
                "Text type": cols[3].text.strip(),
                "Text number": cols[4].text.strip(),
                "Text title": title,
                "Agenda item": cols[6].text.strip(),
                "Type": cols[7].text.strip(),
                "Topic": cols[8].text.strip(),
                "Main sponsors": cols[9].text.strip(),
                "Means of adoption": cols[10].text.strip(),
                "PBI": cols[11].text.strip(),
                "New Resource Requirements": cols[12].text.strip(),
                "Location": link,
                "Session starting date": cols[14].text.strip(),
                "stringified text-number": cols[15].text.strip(),
            }

        except Exception as exc:
            logger.warning("Skipping row: %s", exc)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with UniversalRightsScraper() as scraper:
        result_df = scraper.scrape()

    result_df.to_csv("ohchr_resolutions.csv", index=False)
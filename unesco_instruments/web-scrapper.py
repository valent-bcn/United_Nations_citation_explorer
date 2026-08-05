import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


class UnescoScraper:
    def __init__(self, headless=True, timeout=30):
        self.headless = headless
        self.timeout = timeout

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
        url = "https://www.unesco.org/en/legal-affairs/list"

        driver = self.make_driver()

        try:
            driver.get(url)

            # Wait until the results are loaded
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.sdh-results a.teaser")
                )
            )

            records = []

            items = driver.find_elements(
                By.CSS_SELECTOR,
                "div.sdh-results a.teaser"
            )

            for item in items:

                # Title
                title = item.find_element(
                    By.CSS_SELECTOR,
                    ".h5"
                ).text.strip()

                # URL
                link = item.get_attribute("href")

                # Location
                try:
                    location = item.find_element(
                        By.CSS_SELECTOR,
                        ".address-wrapper"
                    ).text.strip()
                except:
                    location = ""

                # Date
                try:
                    date = item.find_element(
                        By.CSS_SELECTOR,
                        ".date"
                    ).text.strip()
                except:
                    date = ""

                records.append({
                    "title": title,
                    "date": date,
                    "location": location,
                    "url": link,
                })

            return pd.DataFrame(records)

        finally:
            driver.quit()


if __name__ == "__main__":
    scraper = UnescoScraper(headless=False)

    df = scraper.scrape()

    print(df)

    df.to_csv("unesco_legal_instruments.csv", index=False)
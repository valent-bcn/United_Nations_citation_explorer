from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import pandas as pd
import time

#Instead of scrapping the 'next page' button we follow the link of pages
base_url = "https://www.ohchr.org/en/instruments-listings?page={page}#tab-2"

driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

data = []

# pages 0 to 3 (4 pages total)
for page in range(4):
    url = base_url.format(page=page)
    print(f"\nLoading page {page}: {url}")

    driver.get(url)
    time.sleep(5)

    cards = driver.find_elements(By.CSS_SELECTOR, "a.card-headline-with-eyebrows__link")

    print(f"Found {len(cards)} cards")

    for i, card in enumerate(cards):
        try:
            title = card.find_element(By.CSS_SELECTOR, "h3.card__heading span").text.strip()
            adopted = card.find_element(By.CSS_SELECTOR, "p.text--eyebrow--large").text.strip()
            link = card.get_attribute("href")

            data.append({
                "title": title,
                "adopted_text": adopted,
                "url": link
            })

            print(f"[Page {page} - {i}] {title}")

        except Exception as e:
            print("Error:", e)

driver.quit()

df = pd.DataFrame(data)
print("\nDone:", len(df), "records")

# Optional save
df.to_csv("./ohchr_instruments-universal.csv", index=False)
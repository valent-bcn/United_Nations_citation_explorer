from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

url = "https://www.ohchr.org/en/instruments-listings#tab-1"

driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
driver.get(url)

wait = WebDriverWait(driver, 15)

data = []
page = 1

while True:
    print(f"\nScraping page {page}...")

    # wait until cards are loaded
    wait.until(EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, "a.card-headline-with-eyebrows__link")
    ))

    cards = driver.find_elements(By.CSS_SELECTOR, "a.card-headline-with-eyebrows__link")

    for card in cards:
        try:
            title = card.find_element(By.CSS_SELECTOR, "h3.card__heading span").text.strip()
            adopted = card.find_element(By.CSS_SELECTOR, "p.text--eyebrow--large").text.strip()
            link = card.get_attribute("href")

            data.append({
                "title": title,
                "adopted_text": adopted,
                "url": link
            })

        except Exception as e:
            print("Error:", e)

    # Try to click "Next page"
    try:
        next_button = driver.find_element(By.XPATH, "//a[contains(text(),'Next')]")

        # If disabled, break
        if "disabled" in next_button.get_attribute("class"):
            break

        driver.execute_script("arguments[0].click();", next_button)
        page += 1

    except Exception:
        print("No next button found. Stopping.")
        break

driver.quit()

df = pd.DataFrame(data)
print(df.head())
df.to_csv("./ohchr_instruments-core.csv", index=False)
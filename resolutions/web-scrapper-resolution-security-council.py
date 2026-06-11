import csv
import requests
from bs4 import BeautifulSoup

# ----------------------------
# SETTINGS
# ----------------------------
URL = "https://main.un.org/securitycouncil/en/content/resolutions-0"
OUTPUT_CSV = "./sc_resolutions_1946_2026.csv"
REQUEST_TIMEOUT = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ----------------------------
# HELPERS
# ----------------------------
def fetch_page(url: str) -> BeautifulSoup:
    print(f"Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    print(f"  Final URL : {resp.url}")
    print(f"  Status    : {resp.status_code}")
    return BeautifulSoup(resp.text, "html.parser")


def parse_resolutions(soup: BeautifulSoup) -> list[dict]:
    """
    Walk every accordion section and extract resolution code + title.

    DOM structure:
        <dl class="ckeditor-accordion">
          <dt>
            <a class="ckeditor-accordion-toggler">1961</a>
          </dt>
          <dd class="active" style="display: block;">
            <table class="table table-striped">
              <tr>
                <td><a href="https://undocs.org/S/RES/170(1961)">S/RES/170 (1961)</a></td>
                <td>Admission of new Members …</td>
              </tr>
              ...
            </table>
          </dd>
        </dl>

    All <dd> blocks are present in the HTML regardless of which year is
    currently "open" — the JS only toggles display:none/block at runtime.
    """
    results = []

    for dt in soup.find_all("dt"):
        year = dt.get_text(strip=True)
        print("Year:", year)

        dd = dt.find_next_sibling("dd")

        #print("last test", dd.prettify()[:1000])
        if dd:
            table = dd.find("table")

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            elif len(cells) == 2:
                code_cell, title_cell = cells[0], cells[1]
                adoption = ""

            elif len(cells) == 3:
                code_cell, adoption_cell, title_cell = cells[0], cells[1], cells[2]
                adoption = adoption_cell.get_text(strip=True)
            else:
                continue


            link = code_cell.find("a")

            resolution_code = link.get_text(strip=True) if link else code_cell.get_text(strip=True)
            resolution_url  = link["href"].strip() if link else ""
            title           = title_cell.get_text(strip=True)

            if resolution_code:
                results.append({
                    "year":  year,
                    "code":  resolution_code,
                    "adoption": adoption,
                    "title": title,
                    "url":   resolution_url,
                })

    return results


def save_csv(records: list[dict], path: str) -> None:
    """Write results to a CSV file."""
    fieldnames = ["year", "code", "adoption", "title", "url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} resolutions → {path}")

# ----------------------------
def main():
    soup = fetch_page(URL)
    print(len(soup.find_all("dt")))
    print(soup.find_all("dt")[:5])


    resolutions = parse_resolutions(soup)

    if not resolutions:
        print("\nNo resolutions found.")
        # Debug soup = fetch_page(URL) and inspect the HTML
        return

    # Preview
    print(f"\nFound {len(resolutions)} resolutions total. Sample (first 5):")
    for r in resolutions[:5]:
        print(f"  [{r['year']}] {r['code']}  —  {r['title']}")

    save_csv(resolutions, OUTPUT_CSV)


if __name__ == "__main__":
    main()
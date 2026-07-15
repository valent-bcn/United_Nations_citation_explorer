import csv
import re
import requests
from bs4 import BeautifulSoup

# ----------------------------
# SETTINGS
# ----------------------------
URL = "https://en.wikipedia.org/wiki/List_of_treaties"
BASE_URL = "https://en.wikipedia.org"
OUTPUT_CSV = "./wikipedia_treaties-all.csv"
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


def clean_text(text: str) -> str:
    """Collapse whitespace and strip footnote brackets like [1] or [a] left behind."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_footnotes(soup: BeautifulSoup) -> dict[str, str]:
    """
    Build a mapping like

        cite_note-344 -> full note text
    """
    notes = {}

    for li in soup.select("li[id^='cite_note-']"):
        note_id = str(li.get("id"))

        # Remove the little ↑ backlink
        for backlink in li.select(".mw-cite-backlink"):
            backlink.decompose()

        text = clean_text(li.get_text(" ", strip=True))

        notes[note_id] = text

    return notes

def parse_name_cell(cell, footnotes) -> tuple[str, str, str]:
    """
    Extract (name, url, note) from a "Name" table cell.

    DOM structure (typical):
        <td>
          <a href="/wiki/Treaty_of_Versailles">Treaty of Versailles</a>
          <sup class="reference">[a]</sup>   <- footnote marker, dropped
        </td>

    Sometimes the cell has no link at all (a few very old / disputed
    entries are plain text), and sometimes there is explanatory text in
    the cell alongside the link, e.g. "(unratified)" — that extra text is
    treated as a note.
    """
    # Work on a copy so we can strip footnote markers (<sup class="reference">)
    # without touching the original soup tree used elsewhere.

    cell_copy = BeautifulSoup(str(cell), "html.parser")

    link = cell_copy.find("a", href=re.compile(r"^/wiki/"))

    if link:
        #name = clean_text(link.get_text(" ", strip=True))
        href = link.get("href", "").strip()
        if href.startswith("/wiki/"):
            url = BASE_URL + href
        elif href.startswith("http"):
            url = href
        else:
            url = ""

        name = clean_text(cell_copy.get_text(" ", strip=True))
        note = ""

        # Look for footnote reference
        ref = cell_copy.select_one("sup.reference a[href^='#cite_note-']")
        if ref:
            note_id = ref["href"].lstrip("#")
            note = footnotes.get(note_id, "")

        return name, url, note

    # No link found — the whole cell is the name, no separate note.
    name = clean_text(cell_copy.get_text(" ", strip=True))
    return name, "", ""


def parse_treaties(soup: BeautifulSoup, footnotes: dict) -> list[dict]:
    """
    Walk every wikitable on the page and extract one record per treaty row.

    DOM structure (typical section table):
        <table class="wikitable">
          <tr><th>Year</th><th>Name</th><th>Summary</th></tr>
          <tr>
            <td rowspan="2">1668</td>
            <td><a href="/wiki/...">First Triple Alliance</a></td>
            <td>Alliance between England, the United Provinces and Sweden.</td>
          </tr>
          <tr>
            <!-- no Year <td> here: it belongs to the row above via rowspan -->
            <td><a href="/wiki/...">Treaty of Aix-la-Chapelle (1668)</a></td>
            <td>Ends the War of Devolution ...</td>
          </tr>
          ...
        </table>

    When several treaties share the same year, Wikipedia only renders the
    Year cell once (with a rowspan) on the first of those rows. Every
    following row for that same year omits the Year <td> entirely, so we
    must remember the last seen year and reuse it until the rowspan count
    runs out.
    """
    results = []

    tables = soup.find_all("table", class_="wikitable")
    print(f"Found {len(tables)} wikitable(s) on the page.")

    current_year = ""
    year_rowspan_remaining = 0

    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue  # header row

            # Decide whether this row carries its own Year cell.
            # Heuristic: a genuine Year cell is short (no <a> tag, mostly
            # digits/letters like "c. 1310 BCE" or "1289–1290").
            first_cell = cells[0]
            looks_like_year = (
                first_cell.find("a") is None
                and re.search(r"\d", first_cell.get_text())
                and len(cells) >= 3
            )

            if looks_like_year and year_rowspan_remaining <= 0:
                year_cell, name_cell, summary_cell = cells[0], cells[1], cells[2]
                current_year = clean_text(year_cell.get_text(" ", strip=True))
                rowspan_attr = year_cell.get("rowspan")
                year_rowspan_remaining = int(rowspan_attr) if rowspan_attr else 1
            elif len(cells) >= 2:
                # Continuation row: year is inherited, only Name + Summary present
                name_cell, summary_cell = cells[0], cells[1]
            else:
                continue

            year_rowspan_remaining -= 1

            name, url, note = parse_name_cell(name_cell, footnotes)
            summary = clean_text(summary_cell.get_text(" ", strip=True))

            if not name:
                continue

            results.append({
                "year": current_year,
                "name": name,
                "url": url,
                "note": note,
                "summary": summary,
            })

    return results


def save_csv(records: list[dict], path: str) -> None:
    """Write results to a CSV file."""
    fieldnames = ["year", "name", "url", "note", "summary"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} treaties → {path}")


# ----------------------------
# Fun fact
# Treaty of Hubertusburg
# First treatu of Paris, share the same summary
# ----------------------------
def main():
    soup = fetch_page(URL)

    footnotes = parse_footnotes(soup)

    treaties = parse_treaties(soup, footnotes)

    if not treaties:
        print("\nNo treaties found.")
        # Debug: inspect soup.find_all("table", class_="wikitable") manually
        return

    # Preview
    print(f"\nFound {len(treaties)} treaties total. Sample (first 5):")
    for t in treaties[:5]:
        print(f"  [{t['year']}] {t['name']}  —  {t['summary'][:60]}")

    save_csv(treaties, OUTPUT_CSV)


if __name__ == "__main__":
    main()
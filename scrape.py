import asyncio
import json
import requests
import time
from playwright.async_api import async_playwright

def geocode(address):
    """Geocode using OpenStreetMap Nominatim API."""
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': address + ', Hamilton, ON',
        'format': 'json',
        'limit': 1
    }
    response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
    data = response.json()
    if data:
        return {"lat": float(data[0]['lat']), "lon": float(data[0]['lon'])}
    return None

async def scrape_hoodq():
    schools = []
    categories = {
        "Public": "Public School",
        "Catholic": "Catholic School",
        "Private": "Private School",
        "Alternative/Special": "Alternative/Special School"
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, executable_path="/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta")
        page = await browser.new_page()
        await page.goto("https://www.hoodq.com/explore/hamilton-on/ainslie-wood")
        await page.wait_for_timeout(5000)  # Wait for JS to load

        for section_title, category in categories.items():
            try:
                # Find the section by heading text
                section = await page.query_selector(f"//h3[contains(text(), '{section_title}')]")
                if not section:
                    continue
                ul = await section.evaluate_handle("el => el.nextElementSibling")
                if not ul:
                    continue
                lis = await ul.query_selector_all("li")
                for li in lis:
                    name = (await li.inner_text()).split('\n')[0].strip()
                    if name:
                        schools.append({
                            "name": name,
                            "category": category
                        })
            except Exception:
                continue

        await browser.close()
    return schools

async def main():
    schools = await scrape_hoodq()
    for school in schools:
        geo = geocode(school["name"])
        time.sleep(1)  # Be polite to the API
        school["geo_location"] = geo

    # Output as JSON
    json_output = json.dumps(schools, indent=2)
    print(json_output)

    # Optionally, save to file
    with open('ainslie_wood_schools.json', 'w') as f:
        f.write(json_output)

if __name__ == "__main__":
    asyncio.run(main())

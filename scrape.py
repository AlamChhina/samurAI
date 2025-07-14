import asyncio
import json
import requests
import time
from playwright.async_api import async_playwright

APPLE_MAPS_JWT = "YOUR_APPLE_MAPS_JWT"  # Replace with your JWT

def geocode_apple_maps(address):
    """Geocode using Apple Maps Server API."""
    url = "https://maps-api.apple.com/v1/geocode"
    headers = {
        "Authorization": f"Bearer {APPLE_MAPS_JWT}"
    }
    params = {
        "q": address + ", Hamilton, ON",
        "limit": 1,
        "lang": "en"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        # Apple Maps API returns results in 'results', each with 'coordinate'
        results = data.get("results", [])
        if results and "coordinate" in results[0]:
            coord = results[0]["coordinate"]
            return {"lat": coord["latitude"], "lon": coord["longitude"]}
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

        # Grab text from all key insights divs
        key_insights_divs = await page.query_selector_all('div[class*="hqtw-flex"][class*="hqtw-flex-col"][class*="hqtw-w-full"][class*="hqtw-gap-"]')
        insights = []
        for div in key_insights_divs:
            # Get the heading (span) and the paragraph (p)
            heading = await div.query_selector('span.hqtw-uppercase')
            heading_text = await heading.inner_text() if heading else None
            p = await div.query_selector('p.hqtw-text-black')
            p_text = await p.inner_text() if p else None
            if heading_text and p_text:
                insights.append({"title": heading_text.strip(), "text": p_text.strip()})

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
    return schools, insights

async def main():
    schools, insights = await scrape_hoodq()
    for school in schools:
        geo = geocode_apple_maps(school["name"])
        time.sleep(1)  # Be polite to the API
        school["geo_location"] = geo

    # Output as JSON
    json_output = json.dumps({"schools": schools, "insights": insights}, indent=2)
    print(json_output)

    # Optionally, save to file
    with open('ainslie_wood_schools.json', 'w') as f:
        f.write(json_output)

if __name__ == "__main__":
    asyncio.run(main())

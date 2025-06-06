import requests
from bs4 import BeautifulSoup
import sys

try:
    # Get arguments from command line
    if len(sys.argv) < 2:
        raise IndexError("Not enough arguments provided")
    
    word = sys.argv[1]
    url = sys.argv[2]
except IndexError:
    print("Error: Please provide a word to search and optionally a URL.")
    sys.exit(1)

# Send HTTP GET request
response = requests.get(url)

if response.status_code == 200:
    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Collect all matching <li> tags grouped by 14-inch and 16-inch MacBook
    matching_16_inch = []

    for li_tag in soup.find_all('li'):
        if word in li_tag.text and 'Apple M4 ' in li_tag.text and '16-inch MacBook' in li_tag.text:
            h3_tag = li_tag.find('h3')
            a_tag = h3_tag.find('a') if h3_tag else None
            title = a_tag.text.strip() if a_tag else ''
            link = a_tag['href'] if a_tag and a_tag.has_attr('href') else None
            if link and link.startswith('/'):
                link = f"https://www.apple.com{link}"
            price_tag = li_tag.find('div', class_='as-price-currentprice as-producttile-currentprice')
            price = price_tag.get_text(strip=True) if price_tag else ''
            # Remove the literal string '<span class="visuallyhidden">Now</span>' if present
            price = price.replace('<span class="visuallyhidden">Now</span>', '').strip()
            # Remove 'Now' if present (from any other source)
            price = price.replace('Now', '').strip()
            # Extract numeric value from price string (e.g., $4,499.00 -> 4499.00)
            import re
            price_value = None
            price_match = re.search(r'\$([\d,]+\.\d{2})', price)
            if price_match:
                price_value = float(price_match.group(1).replace(',', ''))
            if link and link.startswith('https://') and price_value is not None and price_value < 4999.00:
                matching_16_inch.append(f"{title} | {price}\n{link}\n")

    # Print grouped matching <li> tags at the end
    if matching_16_inch:
        print(f"Found '{word}':")
        print("\n16-inch MacBook:\n")
        for li in matching_16_inch:
            print(f"{li}")
    else:
        print(f"\n'{word}' not in {url}.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")


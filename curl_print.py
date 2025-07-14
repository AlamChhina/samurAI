import sys
import requests
from bs4 import BeautifulSoup

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        sys.exit(1)

    soup = BeautifulSoup(response.text, 'html.parser')
    ps = soup.find_all('p')

    # print(f"Found {len(ps)} <p> elements.\n")

    for i, div in enumerate(ps, 1):
        text = div.get_text(strip=True)
        if text:  # Print only if there is text
            print(text)
            print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()

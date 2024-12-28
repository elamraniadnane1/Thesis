import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the base URL and pagination pattern
base_url = 'https://www.hespress.com/politique'
pagination_url = 'https://www.hespress.com/politique/page/{}/'


# Set headers to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/90.0.4430.93 Safari/537.36'
}

# Define the number of pages you want to scrape
num_pages = 25  # Adjust as needed

# Initialize a list to store all political URLs
all_politics_urls = []

for page in range(1, num_pages + 1):
    if page == 1:
        url = base_url
    else:
        url = pagination_url.format(page)
    
    logging.info(f"Scraping Page {page}: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for Page {page}: {e}")
        continue
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all divs with class 'card-img-top'
    card_divs = soup.find_all('div', class_='card-img-top')
    
    # Iterate through each card div to find political articles
    for div in card_divs:
        # Check if the span with class 'cat politique' exists
        category_span = div.find('span', class_='cat politique')
        if category_span:
            # Find the <a> tag with class 'stretched-link'
            a_tag = div.find('a', class_='stretched-link')
            if a_tag and 'href' in a_tag.attrs:
                article_url = a_tag['href']
                article_title = a_tag.get('title', '').strip()
                all_politics_urls.append({'title': article_title, 'url': article_url})
                logging.info(f"Extracted URL: {article_url}")
    
    # Random sleep to mimic human browsing and avoid being blocked
    sleep_time = random.uniform(1, 3)
    logging.info(f"Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

# Remove duplicates (if any)
unique_politics_urls = [dict(t) for t in {tuple(d.items()) for d in all_politics_urls}]

# Display the extracted URLs
logging.info(f"\nTotal Extracted URLs: {len(unique_politics_urls)}")
for idx, article in enumerate(unique_politics_urls, start=1):
    print(f"{idx}. {article['title']}: {article['url']}")

# Save to CSV
if unique_politics_urls:
    try:
        df = pd.DataFrame(unique_politics_urls)
        df.to_csv('hespress_politics_urls.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Saved all extracted URLs to hespress_politics_urls.csv")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
else:
    logging.info("No political article URLs were extracted.")

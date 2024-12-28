import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# URL of the Hespress article
url = 'https://www.hespress.com/%d8%ad%d9%83%d9%8a%d9%85%d9%8a-%d9%88%d8%a7%d9%84%d9%83%d8%b9%d8%a8%d9%8a-%d9%8a%d8%aa%d9%85%d9%8a%d8%b2%d8%a7%d9%86-%d9%81%d9%8a-2024-1489088.html'

# Set up Selenium WebDriver with options
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in headless mode
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')  # Bypass OS security model
options.add_argument('--window-size=1920,1080')  # Set window size to ensure all elements load

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def load_all_comments(driver):
    """
    Function to click the "Load More" button until all comments are loaded.
    Adjust the XPath based on the actual button on the webpage.
    """
    while True:
        try:
            # Example XPath for "Load More" button; adjust as needed
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More') or contains(text(), 'تحميل المزيد')]"))
            )
            load_more_button.click()
            logging.info("Clicked 'Load More' button to load more comments.")
            time.sleep(random.uniform(1, 3))  # Wait for new comments to load
        except:
            # No more "Load More" buttons found
            logging.info("No more 'Load More' buttons found. All comments loaded.")
            break

try:
    # Open the URL
    driver.get(url)
    logging.info(f"Opened URL: {url}")

    # Wait for the main content to load
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))  # Adjust selector as needed

    # Optionally, load all comments if there's a "Load More" button
    load_all_comments(driver)

    # Parse the rendered HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract Article Title
    try:
        title = soup.find('h1').get_text(strip=True)
        logging.info(f"Title: {title}\n")
    except AttributeError:
        logging.error("Error: Could not find the article title.")
        title = None

    # Extract Publication Date
    try:
        # Define multiple possible selectors for publication date
        pub_date = None
        possible_selectors = [
            {'tag': 'span', 'class': 'publication-date'},
            {'tag': 'div', 'class': 'article-meta'},
            {'tag': 'time', 'class': 'published'},
            {'tag': 'div', 'class': 'article-date'},  # Original selector
        ]

        for selector in possible_selectors:
            element = soup.find(selector['tag'], class_=selector['class'])
            if element:
                pub_date = element.get_text(strip=True)
                break

        if pub_date:
            logging.info(f"Publication Date: {pub_date}\n")
        else:
            logging.error("Error: Could not find the publication date.")
    except AttributeError:
        logging.error("Error: Issue extracting the publication date.")
        pub_date = None

    # Extract Article Content
    try:
        # Adjust the class name based on actual webpage
        content_div = soup.find('div', class_='content-article')
        if not content_div:
            # Try alternative selector
            content_div = soup.find('div', class_='article-content')
        if content_div:
            paragraphs = content_div.find_all('p')
            content = '\n'.join([para.get_text(strip=True) for para in paragraphs])
            logging.info(f"Content: \n{content}\n")
        else:
            logging.error("Error: Could not find the article content.")
            content = None
    except AttributeError:
        logging.error("Error: Issue extracting the article content.")
        content = None

    # Extract Images (Optional)
    try:
        images = content_div.find_all('img')
        image_urls = [img['src'] for img in images if 'src' in img.attrs]
        logging.info(f"Images: {image_urls}\n")
    except AttributeError:
        logging.error("Error: Could not extract images.")

    # Extract Comments
    comments = []
    try:
        # Adjust the class name based on actual webpage
        comments_section = soup.find('div', class_='comments-area')
        if comments_section:
            # Find the <ul> containing all comments
            comment_list = comments_section.find('ul', class_='comment-list')
            if comment_list:
                comment_divs = comment_list.find_all('li', class_='comment')
                for idx, comment in enumerate(comment_divs, start=1):
                    try:
                        # Extract commenter name
                        commenter = comment.find('span', class_='fn').get_text(strip=True)
                        # Extract comment date
                        comment_date = comment.find('div', class_='comment-date').get_text(strip=True)
                        # Extract comment text
                        comment_text = comment.find('div', class_='comment-text').get_text(strip=True)
                        comments.append({
                            'commenter': commenter,
                            'comment_date': comment_date,
                            'comment': comment_text
                        })
                        logging.info(f"Extracted Comment {idx} by {commenter}")
                    except AttributeError:
                        logging.warning(f"Warning: Incomplete data for comment {idx}. Skipping.")
                        continue
                if comments:
                    logging.info(f"Extracted {len(comments)} comments.\n")
                else:
                    logging.info("No comments extracted.")
            else:
                logging.info("No comment list found within the comments section.")
        else:
            logging.info("No comments section found in the HTML.")
    except Exception as e:
        logging.error(f"Error extracting comments: {e}")

finally:
    # Close the browser
    driver.quit()
    logging.info("Browser closed.")

# Save Comments to CSV (Optional)
if comments:
    try:
        df = pd.DataFrame(comments)
        df.to_csv('comments.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(comments)} comments to comments.csv")
    except Exception as e:
        logging.error(f"Error saving comments to CSV: {e}")
else:
    logging.info("No comments to save.")

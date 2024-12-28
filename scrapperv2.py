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

# Function to initialize Selenium WebDriver
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')  # Bypass OS security model
    options.add_argument('--window-size=1920,1080')  # Set window size to ensure all elements load
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Function to load all comments by clicking "Load More" buttons if present
def load_all_comments(driver):
    while True:
        try:
            # Adjust the XPath based on the actual button text or attributes
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More') or contains(text(), 'تحميل المزيد')]"))
            )
            load_more_button.click()
            logging.info("Clicked 'Load More' button to load more comments.")
            time.sleep(random.uniform(1, 3))  # Wait for new comments to load
        except Exception:
            # No more "Load More" buttons found
            logging.info("No more 'Load More' buttons found. All comments loaded.")
            break

# Function to extract article details and comments from a single URL
def extract_article_data(driver, url):
    article_data = {
        'title': None,
        'url': url,
        'publication_date': None,
        'content': None,
        'images': [],
    }
    comments_data = []

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
            article_data['title'] = title
            logging.info(f"Title: {title}\n")
        except AttributeError:
            logging.error("Error: Could not find the article title.")

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
                article_data['publication_date'] = pub_date
                logging.info(f"Publication Date: {pub_date}\n")
            else:
                logging.error("Error: Could not find the publication date.")
        except AttributeError:
            logging.error("Error: Issue extracting the publication date.")

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
                article_data['content'] = content
                logging.info(f"Content Extracted.\n")
            else:
                logging.error("Error: Could not find the article content.")
        except AttributeError:
            logging.error("Error: Issue extracting the article content.")

        # Extract Images (Optional)
        try:
            images = content_div.find_all('img') if content_div else []
            image_urls = [img['src'] for img in images if 'src' in img.attrs]
            article_data['images'] = image_urls
            logging.info(f"Images Extracted: {len(image_urls)} images found.\n")
        except AttributeError:
            logging.error("Error: Could not extract images.")

        # Extract Comments
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
                            comments_data.append({
                                'article_url': url,
                                'commenter': commenter,
                                'comment_date': comment_date,
                                'comment': comment_text
                            })
                            logging.info(f"Extracted Comment {idx} by {commenter}")
                        except AttributeError:
                            logging.warning(f"Warning: Incomplete data for comment {idx}. Skipping.")
                            continue
                    if comments_data:
                        logging.info(f"Extracted {len(comments_data)} comments.\n")
                    else:
                        logging.info("No comments extracted.")
                else:
                    logging.info("No comment list found within the comments section.")
            else:
                logging.info("No comments section found in the HTML.")
        except Exception as e:
            logging.error(f"Error extracting comments: {e}")

    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}")

    return article_data, comments_data

def main():
    # Read the CSV file containing titles and URLs
    try:
        df_urls = pd.read_csv('hespress_politics_urls.csv')
        logging.info(f"Loaded {len(df_urls)} URLs from hespress_politics_urls.csv")
    except FileNotFoundError:
        logging.error("Error: hespress_politics_urls.csv not found.")
        return
    except Exception as e:
        logging.error(f"Error reading hespress_politics_urls.csv: {e}")
        return

    # Initialize Selenium WebDriver
    driver = init_driver()

    # Initialize lists to store all data
    all_articles = []
    all_comments = []

    # Iterate through each URL in the CSV
    for index, row in df_urls.iterrows():
        title_csv = row['title']
        url = row['url']
        logging.info(f"Processing Article {index + 1}: {title_csv}")

        article_data, comments_data = extract_article_data(driver, url)

        # Append article data
        all_articles.append(article_data)

        # Append comments data
        all_comments.extend(comments_data)

        # Random sleep between processing articles to mimic human behavior
        sleep_time = random.uniform(2, 5)
        logging.info(f"Sleeping for {sleep_time:.2f} seconds before next article...")
        time.sleep(sleep_time)

    # Close the Selenium WebDriver
    driver.quit()
    logging.info("Browser closed.")

    # Save all articles to a CSV file
    if all_articles:
        try:
            df_articles = pd.DataFrame(all_articles)
            df_articles.to_csv('hespress_politics_details.csv', index=False, encoding='utf-8-sig')
            logging.info(f"Saved {len(df_articles)} articles to hespress_politics_details.csv")
        except Exception as e:
            logging.error(f"Error saving articles to CSV: {e}")
    else:
        logging.info("No article data to save.")

    # Save all comments to a separate CSV file
    if all_comments:
        try:
            df_comments = pd.DataFrame(all_comments)
            df_comments.to_csv('hespress_politics_comments.csv', index=False, encoding='utf-8-sig')
            logging.info(f"Saved {len(df_comments)} comments to hespress_politics_comments.csv")
        except Exception as e:
            logging.error(f"Error saving comments to CSV: {e}")
    else:
        logging.info("No comments data to save.")

if __name__ == "__main__":
    main()

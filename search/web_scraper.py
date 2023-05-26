"""Selenium web scraping module."""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver


from config import Config
import re
import requests

FILE_DIR = Path(__file__).parent.parent
CFG = Config()
KEYWORDS = ['about us', 'about', 'services', 'our services', 'products', 'our products']
KEYWORD_PATTERN = re.compile('|'.join(KEYWORDS), re.IGNORECASE)
SESSION = requests.Session()


def extract_metadata_selenium(url):
    meta_description = ""
    meta_keywords = ""
    options = webdriver.ChromeOptions()

    options.add_argument("--headless")
    options.add_argument("--disable-javascript")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    page_content = driver.page_source
    soup = BeautifulSoup(page_content, "html.parser")

    meta_tags = soup.find_all("meta")
    for meta_tag in meta_tags:
        if meta_tag.get("name", "").lower() == "description":
            meta_description = meta_tag.get("content")
            print(meta_description)
        elif meta_tag.get("name","").lower() == "keywords":
            meta_keywords = meta_tag.get("content")
            print(meta_keywords)

    return meta_description, meta_keywords


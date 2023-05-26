from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup
from search.google_search import get_ddg_summary, google_search, google_official_search
from flask import Flask, request, jsonify, render_template
# from search.web_scraper import extract_metadata_selenium

app = Flask(__name__)

MODEL_DIR = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)


def extract_metadata_selenium(url):
    meta_description = ""
    meta_keywords = ""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-javascript")

    try:
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

    except WebDriverException as e:
        # print(f"Error accessing URL through WebDriver: {e}")
        return None, None

    finally:
        try:
            driver.quit()
        except UnboundLocalError:
            pass


def get_description(name,country,domain=None):
    results = ""
    if domain.startswith("https://"):
        link = domain
    else:
        link = "https://" + domain
    print(domain)
    ddg_data = get_ddg_summary(f"{name}")
    if domain is not None or domain!="https://":
        meta_data,_ = extract_metadata_selenium(domain)
    else:
        #can add the search module here for further data gathering --IMPORTANT!!
        results = ""
    results = str(ddg_data) + ""+ str(meta_data)
    print(results)
    predictions = ner_pipeline(results)
    PRODS = [str(prediction['word']) for prediction in predictions]
    if len(PRODS)==0:
        return results

    return PRODS




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data['name']
    country = data['country']
    domain = data['domain']

    results = get_description(name, country, domain)

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
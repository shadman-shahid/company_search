# Company Search results API:
> The repo has three main files as given below. Each of these files have their own readme files, appropriately named. Read those for more information. This, the main `README.md` file, has the documentation of the `inference.py` file as well.

1. `inference.py` - for running inference/getting results via API endpoint
2. `train.py` - for training model
3. `prepare_data_for_annotation.py` - for preparaing data for annotation via __doccano__ package. __Doccano__ is a separate package that has to be installed and set up as a server for collaborative annotation.

# Running the `inference.py` file. (Main web API file.)

## The Application

> The application is a Flask web application that leverages a pretrained Natural Language Processing (NLP) model to extract and return key entities from a given text content. 
>
> The application uses a model trained for Named Entity Recognition (NER) tasks to predict and return __Products \& Services__ from text scraped from the web. The application uses Selenium and BeautifulSoup to scrape metadata from web pages.

## Library Dependencies

1. **Transformers** 
2. **Selenium** 
3. **BeautifulSoup** 
4. **search.google_search:** A custom library created for the application to handle Google and DuckDuckGo search related operations.
5. **Flask**

### Important Variables:

1. `app`: This is our Flask application.

2. `MODEL_DIR`: This is the directory path of the trained NLP model.

3. `tokenizer`: This is the tokenizer corresponding to the trained NLP model.

4. `model`: This is the trained NLP model.

5. `ner_pipeline`: This is the pipeline that combines the tokenizer and the model for entity recognition tasks.

### Functions:

1. `extract_metadata_selenium(url)`: This function uses Selenium and BeautifulSoup to scrape the `description` and `keywords` meta tags from a given URL. It returns the content of the `description` and `keywords` meta tags, or `None, None` if there is an exception.

2. `get_description(name, country, domain=None)`: This function gathers data from DuckDuckGo's summary of the search term using `get_ddg_summary` function and the metadata of the provided domain from the `extract_metadata_selenium` function. It concatenates these pieces of information and feeds it to the NER pipeline. The function returns the entities recognized by the NER model.

### API Endpoints:

1. `'/'`: The index endpoint, it renders the 'index.html' template.

2. `'/api/predict'`: This is a POST endpoint that accepts JSON data in the format `{'name': name, 'country': country, 'domain': domain}`. The function uses these data to call the `get_description` function, and returns the recognized entities as JSON.

### Running The Application

To run the application, execute the script. The Flask server will start running on `0.0.0.0` and listen on port `5000`.

## Note

1. Make sure the `'trained_model'` directory exists and contains the pretrained NLP model and its corresponding tokenizer.

2. The custom `google_search` module and its functions (`get_ddg_summary`, `google_search`, `google_official_search`) are imported but not used in the provided code. If they are intended to be used, make sure this module exists and is accessible by the application.

3. The `name` and `country` arguments are accepted by the `get_description` function but are not used in the provided code. If they are intended to be used, include them in the function's logic.

4. Ensure 'index.html' exists in the templates directory of your application for the root route to function correctly.

5. The application needs to be connected with a `ChromeDriver` for __Selenium__ to function properly. Ensure that it is installed and correctly set up.
6. Link to the trained model: https://drive.google.com/file/d/1hPojbaYvZaFTo3r6yvQG4f75g37k71lH/view?usp=share_link

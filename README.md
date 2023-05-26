# 1. Running the `inference.py` file.

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


# 2. Running the `train.py` file.

## The Application

> This application is used for training a Named Entity Recognition (NER) model using the Transformers library.

## Library Dependencies

1. **NLTK** 
2. **torch**
3. **Pandas**
4. **json:** A module in Python that provides a method for encoding and decoding JSON objects.
5. **sklearn**

### Important Variables:

1. `model_checkpoint`: This is the model checkpoint used as the basis for fine-tuning.

2. `tokenizer`: This is the tokenizer corresponding to the model checkpoint.

3. `config`: This is the configuration object corresponding to the model checkpoint.

4. `model`: This is the model used for fine-tuning.

5. `device`: This is the device (CPU or GPU) used for training.

6. `training_args`: These are the arguments used for training.

7. `trainer`: This is the trainer used for fine-tuning the model.

### Functions:

1. `compute_metrics(eval_pred: EvalPrediction)`: This function calculates the precision, recall, and F1 score based on the predicted and true labels.

2. `get_labels(text, labels)`: This function maps entities to tokens in a given sentence.

3. `convert_data_to_ner_format(data)`: This function converts the data into a format suitable for Named Entity Recognition (NER).

### Running The Application

The script loads a dataset from a JSON lines file, converts the data into the format required for Named Entity Recognition (NER), tokenizes the data, and fine-tunes a pre-trained model on this data.

The training script fine-tunes the model for 10 epochs, logging the training loss every 200 steps. The best model is saved based on the evaluation loss.

## Note

1. Ensure the jsonl file `'./data/final_dataset.jsonl'` exists and contains the correct data in a suitable format for Named Entity Recognition. The `final_dataset.jsonl` file is inferred to contain JSON objects, one per line, where each JSON object represents an item in the dataset. 
> The file must have the following structure, or must be converted to the following structure for the code to work. **Note**: This is the standard structure for dataset prepared through **doccano** package, that we have used.
```json
{
    "text": "Amazon is an e-commerce platform. But it also has online streaming services. ",
    "entities": [
        {
            "start_offset": 14,
            "end_offset": 23,
            "label": "PROD"
        },
        ...
    ]
}
```
> This `.jsonl` file can be processed line by line, with each line being a separate JSON object representing a single data point. Each line's JSON object has a `text` field that is a string of text, and an `entities` field that is a list of dictionaries, each containing `start_offset`, `end_offset`, and `label` fields.

2. The `torch.device()` function is used to move the model and data to a GPU, if available. Ensure that your environment has a compatible GPU if you want to use this feature.

3. The model_checkpoint in the script is "distilbert-base-uncased". You can replace this with the path to a different model checkpoint if desired.

4. The training_args object controls the parameters for training. You can customize these parameters as needed.

5. The script uses a basic split of 80-20 for the training and testing datasets.

6. This script trains a NER model. Make sure that the data is labeled in a suitable format for Named Entity Recognition.



# 3. Running `prepare_data_for_annotation.py` file.

## Overview

The `prepare_data_for_annotation.py` script is used to process a large CSV file of company data (`./data/companies_sorted.csv` - obtained from [this kaggle dataset](https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset)), fetch descriptions of each company from the DuckDuckGo search results or from the company's domain using Selenium. This information is then written to a JSON Lines (JSONL) file that can be used for further processing or annotation. 

## Dependencies

- __pandas__
- __duckduckgo_search__
- __concurrent.futures__: Used for parallel processing of chunks.
- __search.web_scraper__: A custom module which contains `extract_metadata_selenium` function to extract metadata from a webpage using `Selenium`.
- __search.google_search__: A custom module which contains `get_ddg_summary` function to get search summary from DuckDuckGo.

## Input

A CSV file located at `./data/companies_sorted.csv` which is expected to have at least two columns: `'name'` and `'domain'`.

## Output

The script produces a JSONL file `./data/json_chunk{NCHUNK}-{chunk_size}.jsonl` with company data in a format suitable for annotation.

## Workflow

1. The script reads the company data file in chunks. The size of each chunk is determined by `CHUNK_SIZE`.

2. For each chunk, the script fetches the description of each company. The source of the description is DuckDuckGo search results. If the search results are not available, the script attempts to extract metadata from the company's website. If both sources fail, the description is set to `None`.

3. The company information including name, domain and fetched description is appended to a list of JSON objects.

4. This processing is parallelized across different chunks using Python's ThreadPoolExecutor. 

5. Once all chunks are processed, the list of JSON objects is converted to a pandas DataFrame and then saved to a JSONL file.

## Important Note

The script processes only up to the chunk number `NCHUNK` for testing or partial processing purposes. To process the whole file, this restriction should be removed.

The comment in the `process_chunk` function mentions adding a search module for further data gathering if the first two methods to get company description fail. This is an area for future improvements.

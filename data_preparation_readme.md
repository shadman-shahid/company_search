# Running `prepare_data_for_annotation.py` file.

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

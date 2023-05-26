import pandas as pd
from duckduckgo_search import ddg, ddg_answers
from search.web_scraper import extract_metadata_selenium
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from search.google_search import get_ddg_summary


CHUNK_SIZE = 1000
NCHUNK = 4
COMPANY_DATA_PATH = './data/companies_sorted.csv'


def process_chunk(chunk):
    json_data = []
    for _, row in chunk.iterrows():
        results = get_ddg_summary(row['name'])
        if results is None:
            if row['domain'] is not None:
                results,_ = extract_metadata_selenium(f"https://{row['domain']}")
            else:
                #can add the search module here for further data gathering --IMPORTANT!!
                results = None

        json_data.append({"name": row['name'], "domain": row['domain'], "description": results})
    return json_data


if __name__=='__main__':
    json_data = []

    # Get the total number of rows
    total_rows = sum(1 for row in open(COMPANY_DATA_PATH))

    # Calculate the total number of chunks
    total_chunks = total_rows // CHUNK_SIZE
    if total_rows % CHUNK_SIZE != 0:
        total_chunks += 1
    print(total_chunks)

    # Read the file again and only process the last two chunks
    with ThreadPoolExecutor() as executor:
        futures = []
        for chunk_no, chunk in tqdm(enumerate(pd.read_csv(COMPANY_DATA_PATH, chunksize=CHUNK_SIZE))):
            if chunk_no == NCHUNK:

              futures.append(executor.submit(process_chunk, chunk))
              break

        for future in futures:
            json_data.extend(future.result())

    df = pd.DataFrame(json_data)
    df.to_json("./data/json_chunk{NCHUNK}-{chunk_size}.jsonl", orient='records', lines=True)

from __future__ import annotations

import json

from duckduckgo_search import ddg, ddg_answers
from config import Config

CFG = Config()


def google_search(query: str, num_results: int = 8, create_json=True) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:
        return json.dumps(search_results)

    for j in results:
        search_results.append(j)

    # if create_json:
    #     return json.dumps(search_results, ensure_ascii=False, indent=4)

    return search_results


def google_official_search(query: str, num_results: int = 8) -> str | list[str]:
    """Return the results of a Google search using the official Google API

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = CFG.google_api_key
        custom_search_engine_id = CFG.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = (
            service.cse()
            .list(q=query, cx=custom_search_engine_id, num=num_results)
            .execute()
        )

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get(
            "code"
        ) == 403 and "invalid API key" in error_details.get("error", {}).get(
            "message", ""
        ):
            return "Error: The provided Google API key is invalid or missing."
        else:
            return f"Error: {e}"

    # Return the list of search result URLs
    return search_results_links


def get_ddg_summary(company, min_length=800, top_n=5):
    results = ddg_answers(keywords=f"{company}",)
    # results is a list of dict.

    # If the list is empty or is less than a specified length, check related posts as well.
    if len(results)==0 or len(results[0]['text']) < min_length:
        # print("Too short! Trying again.")
        text = []
        for i, answer in enumerate(ddg_answers(keywords=f"{company}", related=True)):
            if i > top_n:
                break
            text.append(answer['text'])
        description = " ".join(text)

        # ddg_answers unable to get any results. Falling back to ddg (text search)
        if len(description) == 0:
            # print("Using ddg_text!...")
            results = ddg(keywords=f"How does {company} make money?", safesearch="Off", max_results=top_n)
            print(company, results)
            description = " ".join([result['body'] for result in results])

        return description

    # print("Found in first try!")
    description = results[0]['text']
    return description
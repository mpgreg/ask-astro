from __future__ import annotations

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from include.tasks.utils.html_helpers import get_all_links

def extract_html(source: dict) -> list[pd.DataFrame]:
    """
    This task scrapes docs from a website and returns a list of pandas dataframes. 
    Return type is a list in order to map to upstream dynamic tasks.  The code recursively 
    generates a list of html files relative to the source's 'base_url' and then extracts each as html.

    Any documents with a url matching a regex pattern specified in exclude_docs will not be extracted.

    Note: Only the html classes from the source's container_class and children are extracted.

    param source: A dictionary specifying base_url, docs to exclude and the container_class
     Example
        {
            "base_url": "https://docs.astronomer.io/astro/", 
            "exclude_docs": [r'[1-9]$'], 
            "container_class": "theme-doc-markdown markdown"
        }
    type source: dict

    The returned data includes the following fields:
    'docLink': URL for the page
    'content': HTML content of the page
    """

    all_links = {source['base_url']}
    get_all_links(
        url=source['base_url'],
        url_base=source['base_url'],
        all_links=all_links
        )
    
    all_links = {link for link in all_links if \
                 not any([re.search(pattern, link) for pattern in source['exclude_docs']])}

    df = pd.DataFrame(all_links, columns=["docLink"])

    df["html_content"] = df["docLink"].apply(lambda x: requests.get(x).content)

    df["content"] = df["html_content"].apply(
        lambda x: str(
            BeautifulSoup(x, "html.parser").find("div", class_=source["container_class"]))
        )
    df["content"] = df["content"].apply(lambda x: re.sub("¶", "", x))

    df.drop_duplicates(subset=["docLink"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["content", "docLink"]]

    return [df]

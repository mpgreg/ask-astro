from __future__ import annotations

from bs4 import BeautifulSoup
from include.tasks.utils.html_helpers import get_all_links
import logging
import pandas as pd
import re
import requests

logger = logging.getLogger("airflow.task")

def extract_html(source: dict) -> list[pd.DataFrame]:
    """
    This task scrapes docs from a website and returns a list of pandas dataframes. 
    Return type is a list in order to map to upstream dynamic tasks.  The code recursively 
    generates a list of html files relative to the source's 'base_url' and then extracts each as html.

    Any documents with a url matching a regex pattern specified in exclude_docs will not be extracted.

    Note: Only the html classes from the source's container_class and children are extracted.

    :param: source: A dictionary specifying base_url, docs to exclude and the container_class
     Example
        {
            "base_url": "https://docs.astronomer.io/astro/", 
            "exclude_docs": [r"[1-9]$"], 
            "container_class": "theme-doc-markdown markdown"
        }

    :return: A dataframe
    """

    all_links = {source["base_url"]}
    get_all_links(
        url=source["base_url"],
        url_base=source["base_url"],
        all_links=all_links
        )
    
    all_links = {link for link in all_links if \
                 not any([re.search(pattern, link) for pattern in source["exclude_docs"]])}

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

def extract_html_discourse(source: dict) -> list[pd.DataFrame]:
    """
    This function extracts json from a Discourse server endpoint based on a source definition dictionary.

    The source can specify categories (ie. Site Feedback) to exclude from extration.

    After extracting posts from the discourse site some logic is applied to select "high-quality" posts 
    based on:
    - score: only topics with an initial thread with a score in the top quartile are kept.
    - trust level: For all replies only posts >= the trust_level_cutoff are kept. 

    All posts are grouped by topic and consolidated into a 'content' field.
    
    :param source: A dictionary of base_url, trust_level_cutoff, and exclude_categories. For example
    {
        "base_url": "https://forum.astronomer.io",
        "trust_level_cutoff": 3,
        "exclude_categories": [
            {'name': 'Site Feedback', 'id': 3},
            {'name': 'Nebula', 'id': 6},
            {'name': 'Software', 'id': 5},
            ]
    }
    :return: A pandas dataframe with 'content' and 'docLink'

    """

    post_columns = [
        "id",
        "topic_id",
        # "topic_slug",
        "score",
        "cooked",
        "user_deleted",
        # "accepted_answer",
        # "topic_accepted_answer",
        "post_number",
        "trust_level"
    ]
    
    exclude_category_ids = [cat["id"] for cat in source["exclude_categories"]]

    response = requests.get(source["base_url"]+"/categories.json?include_subcategories=True")
    if response.ok:
        parent_categories = response.json()["category_list"]["categories"]
        parent_category_ids = [
            category["id"] for category in parent_categories if category["id"] not in exclude_category_ids
            ]

    topics = []
    for category in parent_category_ids:
        logger.info(category)
        response = requests.get(source["base_url"]+f"/c/{category}.json")
        if response.ok:
            response = response.json()["topic_list"]
            topics += [
                topic for topic in response["topics"] if topic["category_id"] not in exclude_category_ids
                ]
            while response.get("more_topics_url"):
                next_page = response["more_topics_url"].replace("?", ".json?")
                logger.info(next_page)
                response = requests.get(source["base_url"] + next_page)
                if response.ok:
                    response = response.json()["topic_list"]
                    topics += [
                        topic for topic in response["topics"] if topic["category_id"] not in exclude_category_ids
                        ]

    topic_ids = {topic["id"] for topic in topics}
    
    posts = []
    for topic_id in topic_ids:
        response = requests.get(source["base_url"]+f"/t/{topic_id}.json")
        if response.ok:

            response = response.json()
            post_ids = response["post_stream"]["stream"]
            logger.info(post_ids)
            post_ids_query = "".join([f"post_ids[]={id}&" for id in post_ids])
            
            response = requests.get(source["base_url"] + f"/t/{topic_id}/posts.json?" + post_ids_query)
            if response.ok:
                response = response.json()
                posts += response["post_stream"]["posts"]

    posts_df = pd.DataFrame(posts)[post_columns]

    score_cutoff = posts_df[posts_df.post_number == 1].score.quantile(.75)

    first_posts_df = posts_df[
        (posts_df.post_number==1) & (posts_df.score>=score_cutoff)
        ]
    first_posts_df = first_posts_df[["topic_id", "cooked"]].rename({"cooked":"first_post"}, axis=1)
    
    #remove first post for joining later
    posts_df.drop(posts_df[posts_df.post_number == 1].index, inplace=True)
    
    #select only posts with high trust level
    posts_df.drop(posts_df[posts_df.trust_level<source["trust_level_cutoff"]].index, inplace=True)

    posts_df.sort_values("post_number", inplace=True)
    
    posts_df = posts_df.groupby("topic_id")["cooked"].apply("".join).reset_index()

    posts_df = posts_df.merge(first_posts_df)
    
    posts_df["content"] = posts_df.apply(lambda x: x.first_post + x.cooked, axis=1)
    posts_df["docLink"] = posts_df.topic_id.apply(lambda x: source["base_url"] + f"/t/{x}/1")

    posts_df = posts_df[["content", "docLink"]]

    return [posts_df]

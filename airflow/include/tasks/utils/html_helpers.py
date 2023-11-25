from __future__ import annotations

import urllib.parse
from time import sleep
import requests
from bs4 import BeautifulSoup


def get_links(url: str, url_base: str) -> set:
    """
    Given a HTML url this function scrapes the page for any HTML links (<a> tags) and returns a set of links which:
    a) starts with the same base (ie. scheme + netloc)
    b) is a relative link from the currently read page

    Relative links are converted to absolute links.

    Note: The absolute link may not be unique due to redirects. Need to check for redirects in calling function.
    """
    response = requests.get(url)
    data = response.text
    soup = BeautifulSoup(data, "lxml")

    links = set()
    for link in soup.find_all("a"):
        link_url = link.get("href")

        if link_url.startswith(url_base):
            links.add(link_url)
        elif not link_url.startswith(("http", "#")) and link_url != '/':
            absolute_url = urllib.parse.urljoin(url, link_url)
            if absolute_url.startswith(url_base):
                links.add(absolute_url)

    links = {link for link in links if not link.endswith(('.xml'))}
    links = {link.split('?')[0] for link in links}
    links = {link.split('#')[0] for link in links}
    links = {link for link in links if link.startswith(url_base)}

    return links


def get_all_links(url: str, url_base: str, all_links: set):
    """
    This is a recursive function to find all the sub-pages of a webpage.  Given a starting URL the function
    recurses through all child links referenced in the page.

    The all_links set is updated in recursion so no return set is passed.
    """
    
    links = get_links(url=url, url_base=url_base)

    for link in links:
        # check if the linked page actually exists and get the redirect which is hopefully unique

        response = requests.head(link, allow_redirects=True)
        if response.ok:
            redirect_url = response.url
            if redirect_url not in all_links:
                print(redirect_url)
                all_links.add(redirect_url)
                try:
                    get_all_links(
                        url=redirect_url, 
                        url_base=url_base, 
                        all_links=all_links
                        )
                except Exception as e:
                    print(e)
                    print("Retrying")
                    sleep(5)
                    get_all_links(
                        url=redirect_url, 
                        url_base=url_base,
                        all_links=all_links
                        )

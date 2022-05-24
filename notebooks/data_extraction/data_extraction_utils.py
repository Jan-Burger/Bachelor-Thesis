import pandas as pd
import numpy as np


def data_prep_posts(api, subreddit: str, start_time, end_time, filters: list[str]):

    posts = list(api.search_submissions(
        subreddit=subreddit,   # Subreddit we want to audit
        after=start_time,      # Start date
        before=end_time,       # End date
        filter=filters))       # Column names we want to retrieve

    df = pd.DataFrame(posts)
    df["type"] = "submission"
    return df  # Return dataframe for analysis


def join_submission_title_and_body(title: str, body: str):

    if body != "[removed]":
        return f"{title} {body}"
    else:
        return title


def find_stock_symbols(post: str, stock_list: list[str]):

    post = set(post.split())

    found_stock = list(post.intersection(stock_list))

    if found_stock:

        if "$" in found_stock[0]:
            return found_stock[0].replace("$", "")
        else:
            return found_stock[0]

    return np.nan

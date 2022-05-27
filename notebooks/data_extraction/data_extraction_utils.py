import pandas as pd
import numpy as np


def data_prep_posts(api, subreddit: str, start_time, end_time, filters: list[str]):

    posts = list(api.search_submissions(
        subreddit=subreddit,   # Subreddit we want to audit
        after=start_time,      # Start date
        before=end_time,       # End date
        filter=filters,      # Column names we want to retrieve
        limit=1000000))

    df = pd.DataFrame([obj.d_ for obj in posts])
    df["type"] = "submission"
    return df  # Return dataframe for analysis


def data_prep_comments(api, subreddit: str, start_time, end_time, filters: list[str]):

    posts = list(api.search_comments(
        subreddit=subreddit,   # Subreddit we want to audit
        after=start_time,      # Start date
        before=end_time,       # End date
        filter=filters,      # Column names we want to retrieve
        limit=1000000))

    df = pd.DataFrame([obj.d_ for obj in posts])
    df["type"] = "comment"
    return df  # Return dataframe for analysis


def join_submission_title_and_body(title: str, body: str):

    if body != "[removed]":
        return f"{title} {body}"
    else:
        return title


def find_stock_symbols(post: str, stock_list: list[str]):

    post = set(post.split())

    found_stocks_raw = post.intersection(stock_list)

    if found_stocks_raw:

        # Deduplicate stock with and without "$"
        found_stocks_processed = list({stock.replace("$", "") if "$" in stock else stock for stock in found_stocks_raw})

        return found_stocks_processed

    return np.nan

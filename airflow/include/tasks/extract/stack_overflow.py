from __future__ import annotations

import pandas as pd
from include.tasks.utils.stack_overflow_helpers import (
    process_stack_answers,
    process_stack_answers_api,
    process_stack_comments,
    process_stack_comments_api,
    process_stack_posts,
    process_stack_questions,
    process_stack_questions_api,
)
from stackapi import StackAPI
from weaviate.util import generate_uuid5


def extract_stack_overflow_archive(tag: dict) -> pd.DataFrame:
    """
    This task generates stack overflow documents as a single markdown document per question with associated comments
    and answers. The archive data was pulled from the internet archives and processed to extract tag-related posts.

    :param tag: A dictionary with a Stack Overflow tag name, cutoff_date and links to archived posts and comments.
    :return: A pandas dataframe with 'content' and 'docLink'
    """

    posts_df = pd.concat([pd.read_parquet(url) for url in tag["archive_posts"]], ignore_index=True)

    posts_df = process_stack_posts(posts_df=posts_df, cutoff_date=tag["cutoff_date"])

    comments_df = pd.concat([pd.read_parquet(url) for url in tag["archive_comments"]], ignore_index=True)

    comments_df = process_stack_comments(comments_df=comments_df)

    questions_df = process_stack_questions(posts_df=posts_df, comments_df=comments_df, tag_name=tag["name"])

    answers_df = process_stack_answers(posts_df=posts_df, comments_df=comments_df)

    # Join questions with answers
    df = questions_df.join(answers_df)
    df = df.apply(
        lambda x: pd.Series(
            [x.docLink, "\n".join([x.content, x.answer_text])]
        ),
        axis=1,
    )
    df.columns = ["docLink", "content"]

    df.reset_index(inplace=True, drop=True)

    df.drop_duplicates(subset=["docLink"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["content", "docLink"]]

    return df


def extract_stack_overflow(tag: dict) -> pd.DataFrame:
    """
    This task generates stack overflow documents as a single markdown document per question with associated comments
    and answers.

    :param tag: A dictionary with a Stack Overflow tag name and cutoff_date.
    :return: A pandas dataframe with 'content' and 'docLink'
    """

    SITE = StackAPI(name="stackoverflow", page_size=100, max_pages=1000)

    # https://api.stackexchange.com/docs/read-filter#filters=!-(5KXGCFLp3w9.-7QsAKFqaf5yFPl**9q*_hsHzYGjJGQ6BxnCMvDYijFE&filter=default&run=true
    filter_ = "!-(5KXGCFLp3w9.-7QsAKFqaf5yFPl**9q*_hsHzYGjJGQ6BxnCMvDYijFE"

    questions_dict = SITE.fetch(
        endpoint="questions", tagged=tag["name"], fromdate=tag["cutoff_date"], filter=filter_
    )

    items = questions_dict.pop("items")

    # TODO: check if we need to paginate
    len(items)
    # TODO: add backoff logic.  For now just fail the task if we can't fetch all results due to api rate limits.
    assert not questions_dict["has_more"]

    posts_df = pd.DataFrame(items)
    posts_df = posts_df[posts_df["answer_count"] >= 1]
    posts_df = posts_df[posts_df["score"] >= 1]
    posts_df.reset_index(inplace=True, drop=True)

    # process questions
    questions_df = posts_df
    questions_df["comments"] = questions_df["comments"].fillna("")
    questions_df["question_comments"] = questions_df["comments"].apply(lambda x: process_stack_comments_api(x))
    questions_df = process_stack_questions_api(questions_df=questions_df, tag_name=tag["name"])

    # process associated answers
    answers_df = posts_df.explode("answers").reset_index(drop=True)
    answers_df["comments"] = answers_df["answers"].apply(lambda x: x.get("comments"))
    answers_df["comments"] = answers_df["comments"].fillna("")
    answers_df["answer_comments"] = answers_df["comments"].apply(lambda x: process_stack_comments_api(x))
    answers_df = process_stack_answers_api(answers_df=answers_df)

    # combine questions and answers
    df = questions_df.join(answers_df).reset_index(drop=True)
    df["content"] = df[["question_text", "answer_text"]].apply("\n".join, axis=1)

    df.drop_duplicates(subset=["docLink"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[["content", "docLink"]]

    return df

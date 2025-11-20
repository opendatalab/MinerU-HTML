"""
Prompt generation utilities for LLM inference.

This module provides functions to generate prompts for LLM-based HTML
content classification tasks.
"""

from dripper.base import ITEM_ID_ATTR
from dripper.exceptions import DripperPromptError


def get_full_prompt(html_str: str) -> str:
    """
    Generate full prompt for LLM-based HTML content classification.

    Creates a comprehensive prompt that instructs the LLM to classify HTML
    elements with item IDs as either "main" (primary content) or "other"
    (supplementary content).

    Args:
        html_str: Simplified HTML string with elements containing {ITEM_ID_ATTR}
                 attributes

    Returns:
        Complete prompt string for LLM inference

    Raises:
        DripperPromptError: If prompt generation fails
    """
    try:
        prompt = f"""As a front-end engineering expert in HTML, your task is to analyze the given HTML structure and accurately classify elements with the {ITEM_ID_ATTR} attribute as either "main" (primary content) or "other" (supplementary content). Your goal is to precisely extract the primary content of the page, ensuring that only the most relevant information is labeled as "main" while excluding navigation, metadata, and other non-essential elements.
Guidelines for Classification:
Primary Content ("main")
Elements that constitute the core content of the page should be classified as "main". These typically include:
✅ For Articles, News, and Blogs:
The main text body of the article, blog post, or news content.
Images embedded within the main content that contribute to the article.
✅ For Forums & Discussion Threads:
The original post in the thread.
Replies and discussions that are part of the main conversation.
✅ For Q&A Websites:
The question itself posted by a user.
Answers to the question and replies to answers that contribute to the discussion.
✅ For Other Content-Based Pages:
Any rich text, paragraphs, or media that serve as the primary focus of the page.
Supplementary Content ("other")
Elements that do not contribute to the primary content but serve as navigation, metadata, or supporting information should be classified as "other". These include:
❌ Navigation & UI Elements:
Menus, sidebars, footers, breadcrumbs, and pagination links.
"Skip to content" links and accessibility-related text.
❌ Metadata & User Information:
Article titles, author names, timestamps, and view counts.
Like counts, vote counts, and other engagement metrics.
❌ Advertisements & Promotional Content:
Any section labeled as "Advertisement" or "Sponsored".
Social media sharing buttons, follow prompts, and external links.
❌ Related & Suggested Content:
"Read More", "Next Article", "Trending Topics", and similar sections.
Lists of related articles, tags, and additional recommendations.
Task Instructions:
You will be provided with a simplified HTML structure containing elements with an {ITEM_ID_ATTR} attribute. Your job is to analyze each element's function and determine whether it should be classified as "main" or "other".
Response Format:
Return a JSON object where each key is the {ITEM_ID_ATTR} value, and the corresponding value is either "main" or "other", as in the following example:
{{"1": "other","2": "main","3": "other"}}
🚨 Important Notes:
Do not include any explanations in the output—only return the JSON.
Ensure high accuracy by carefully distinguishing between primary content and supplementary content.
Err on the side of caution—if an element seems uncertain, classify it as "other" unless it clearly belongs to the main content.

Input HTML:
{html_str}

Output format should be a JSON-formatted string representing a dictionary where keys are item_id strings and values are either 'main' or 'other'. Make sure to include ALL item_ids from the input HTML./no_think
"""
    except Exception as e:
        raise DripperPromptError(f'Error in get_full_prompt: {e}')
    return prompt

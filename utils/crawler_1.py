import os
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from langchain_community.llms import HuggingFaceEndpoint


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_HUB_ACCESS_TOKEN not found in .env")


# -----------------------------
# Initialize LLM
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=200
)


# -----------------------------
# Step 1: Extract Links
# -----------------------------
def extract_links(url: str):

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a_tag in soup.find_all("a", href=True):

        full_url = urljoin(url, a_tag["href"])
        link_text = a_tag.get_text(strip=True).lower()

        if link_text:
            links.append((link_text, full_url))

    return links


# -----------------------------
# Step 2: Filter Links
# -----------------------------
def basic_filter_links(links, base_url):

    filtered = []

    base_domain = urlparse(base_url).netloc

    for text, url in links:

        parsed = urlparse(url)

        # keep only same domain
        if parsed.netloc != base_domain:
            continue

        # remove anchors
        if "#" in url:
            continue

        filtered.append((text, url))

    return filtered


# -----------------------------
# Step 3: Format Links for LLM
# -----------------------------
def format_links_for_llm(links):

    formatted = []

    for i, (text, url) in enumerate(links):

        formatted.append({
            "id": i,
            "text": text,
            "url": url
        })

    return formatted


# -----------------------------
# Step 4: LLM Prompt
# -----------------------------
def create_prompt(links_json):

    return f"""
Goal: Find tractor implements.

Below is a list of navigation links from a website.

Which links are most likely to contain agricultural implements?

Return ONLY JSON with this format:

{{
 "urls_to_visit": ["url1", "url2"]
}}

Links:
{json.dumps(links_json, indent=2)}
"""


# -----------------------------
# Step 5: Ask LLM
# -----------------------------
def llm_select_links(links):

    links_json = format_links_for_llm(links)

    prompt = create_prompt(links_json)

    final_prompt = f"<s>[INST] {prompt} [/INST]"

    response = llm.invoke(final_prompt)

    try:

        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        json_text = response[json_start:json_end]

        parsed = json.loads(json_text)

        return parsed["urls_to_visit"]

    except Exception:

        print("LLM Output parsing failed")
        print(response)

        return []


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    homepage = "https://www.tgschmeiser.com/"

    print("Extracting links...")

    links = extract_links(homepage)

    print("Total links:", len(links))

    print("Filtering links...")

    filtered_links = basic_filter_links(links, homepage)

    print("Filtered links:", len(filtered_links))

    print("Sending navigation structure to LLM...")

    next_urls = llm_select_links(filtered_links)

    print("\nLLM Suggested URLs to Visit:\n")

    for url in next_urls:
        print(url)
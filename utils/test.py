from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import json

from langchain_community.llms import HuggingFaceEndpoint


# -----------------------------
# 1. Initialize LLM
# -----------------------------
access_token = "YOUR_HF_TOKEN"

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=access_token,
    temperature=0.1,
    max_new_tokens=300
)


# -----------------------------
# 2. Extract Links
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
# 3. Basic Filtering
# -----------------------------
def basic_filter_links(links, base_url):
    filtered = []
    base_domain = urlparse(base_url).netloc

    for text, url in links:
        parsed = urlparse(url)

        if parsed.netloc != base_domain:
            continue

        if "#" in url:
            continue

        filtered.append((text, url))

    return filtered


# -----------------------------
# 4. Format for LLM
# -----------------------------
def format_links_for_llm(links):
    return [
        {"id": i, "text": text, "url": url}
        for i, (text, url) in enumerate(links)
    ]


# -----------------------------
# 5. Create Prompt
# -----------------------------
def create_filter_prompt(links_json):
    return f"""
Goal: Extract agricultural tractor implements and related product documentation.

From the links below, return only the IDs of links likely to help achieve this goal.

Return JSON only in this format:
{{ "keep_ids": [] }}

Links:
{json.dumps(links_json, indent=2)}
"""


# -----------------------------
# 6. Filter Using LLM
# -----------------------------
def filter_links_with_llm(links):
    formatted_links = format_links_for_llm(links)
    prompt = create_filter_prompt(formatted_links)

    response = llm.invoke(prompt)

    # Extract JSON safely
    json_start = response.find("{")
    json_text = response[json_start:]

    parsed = json.loads(json_text)
    keep_ids = parsed["keep_ids"]

    cleaned_links = [links[i] for i in keep_ids]
    return cleaned_links


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    homepage = "https://www.tgschmeiser.com/"

    print("Extracting links...")
    links = extract_links(homepage)

    print("Basic filtering...")
    links = basic_filter_links(links, homepage)

    print("LLM filtering...")
    filtered_links = filter_links_with_llm(links)

    print("\nRelevant Links:\n")
    for text, url in filtered_links:
        print(f"{text} -> {url}")
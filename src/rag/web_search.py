import os

from dotenv import load_dotenv
from tavily import TavilyClient
from source_filter import filter_trusted_results


# -----------------------------------------
# Load environment variables
# -----------------------------------------

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# -----------------------------------------
# Check API key
# -----------------------------------------

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY not found. "
        "Check your .env file."
    )


# -----------------------------------------
# Tavily Client
# -----------------------------------------

client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# -----------------------------------------
# Trusted medical sources
# -----------------------------------------

TRUSTED_DOMAINS = [
    "cancer.gov",
    "who.int",
    "asco.org",
    "pubmed.ncbi.nlm.nih.gov",
    "nccn.org"
]


# -----------------------------------------
# Web Search Function
# -----------------------------------------

def web_search(query, max_results=5):

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_domains=TRUSTED_DOMAINS
    )

    results = []

    for result in response.get("results", []):

        results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0)
        })


    # -----------------------------------------
    # Filter untrusted sources
    # -----------------------------------------

    trusted_results = filter_trusted_results(
        results
    )

    return trusted_results


# -----------------------------------------
# Test Web Search
# -----------------------------------------

if __name__ == "__main__":

    print("Breast Cancer Web Search")
    print("=" * 60)

    query = input(
        "\nEnter your question: "
    )

    results = web_search(
        query,
        max_results=5
    )

    print("\n")
    print("=" * 60)
    print("TRUSTED WEB SEARCH RESULTS")
    print("=" * 60)

    print(
        "\nTrusted results found:",
        len(results)
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n")
        print(f"RESULT {i}")
        print("-" * 60)

        print(
            "Title:",
            result["title"]
        )

        print(
            "Score:",
            round(result["score"], 4)
        )

        print(
            "URL:",
            result["url"]
        )

        print("\nContent:")

        print(
            result["content"][:1500]
        )
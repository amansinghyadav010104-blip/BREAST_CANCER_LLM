from urllib.parse import urlparse


# -----------------------------------------
# Trusted Medical Domains
# -----------------------------------------

TRUSTED_DOMAINS = {
    "cancer.gov",
    "who.int",
    "asco.org",
    "pubmed.ncbi.nlm.nih.gov",
    "nccn.org"
}


# -----------------------------------------
# Check whether a URL is trusted
# -----------------------------------------

def is_trusted_url(url):
    """
    Return True if the URL belongs to one
    of our approved medical domains.
    """

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        for domain in TRUSTED_DOMAINS:

            if (
                hostname == domain
                or hostname.endswith("." + domain)
            ):
                return True

        return False

    except Exception:
        return False


# -----------------------------------------
# Filter Tavily results
# -----------------------------------------

def filter_trusted_results(results):

    trusted_results = []

    for result in results:

        url = result.get("url", "")

        if is_trusted_url(url):

            trusted_results.append(result)

    return trusted_results


# -----------------------------------------
# Test
# -----------------------------------------

if __name__ == "__main__":

    test_urls = [
        "https://www.cancer.gov/types/breast",
        "https://www.who.int/news-room/fact-sheets/detail/breast-cancer",
        "https://www.asco.org/practice-patients/guidelines",
        "https://pubmed.ncbi.nlm.nih.gov/41213254",
        "https://www.cancertherapyadvisor.com/news/test"
    ]

    print("Trusted Source Filter")
    print("=" * 60)

    for url in test_urls:

        print(
            url,
            "→",
            "TRUSTED" if is_trusted_url(url)
            else "NOT TRUSTED"
        )
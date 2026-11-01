"""Run short, secret-safe Gemini model and generation diagnostics."""

import argparse
import importlib.metadata
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://generativelanguage.googleapis.com/"
DEFAULT_TIMEOUT_SECONDS = 12.0
DIAGNOSTIC_PROMPT = "Reply with exactly: Gemini test successful."


def _error_nodes(error):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "reason", None),
        ))


def classify_network_error(error):
    nodes = list(_error_nodes(error))
    names = " ".join(type(node).__name__.casefold() for node in nodes)
    details = " ".join(str(node).casefold() for node in nodes)
    if any(isinstance(node, ssl.SSLError) for node in nodes) or "ssl" in names:
        return "tls_failure"
    if any(isinstance(node, socket.gaierror) for node in nodes) or any(
        token in details for token in ("name or service not known", "getaddrinfo", "nodename nor servname")
    ):
        return "dns_failure"
    if "proxy" in names or "proxy" in details:
        return "proxy_failure"
    if any(isinstance(node, (TimeoutError, socket.timeout)) for node in nodes) or "timeout" in names:
        return "timeout"
    if any(isinstance(node, (ConnectionError, OSError)) for node in nodes) or any(
        token in details for token in ("connection refused", "connection reset", "network is unreachable")
    ):
        return "network_connection_failure"
    return "network_failure"


def classify_provider_error(error):
    status = get_http_status(error)
    nodes = list(_error_nodes(error))
    details = " ".join(str(node).casefold() for node in nodes)
    if status == 401:
        return "authentication_failure"
    if status == 403:
        return "permission_or_api_access_failure"
    if status == 429:
        return "rate_limit"
    if status in (408, 504):
        return "provider_timeout"
    model_error = any(token in details for token in (
        "model not found", "unknown model", "invalid model", "unsupported model",
        "is not found for api version", "model is not available",
    ))
    if model_error:
        return "invalid_model"
    if status in (400, 404):
        return "invalid_endpoint_or_request"
    if any(
        isinstance(node, (ssl.SSLError, socket.gaierror, TimeoutError, socket.timeout, ConnectionError))
        for node in nodes
    ):
        network_category = classify_network_error(error)
        return "provider_timeout" if network_category == "timeout" else network_category
    if any(isinstance(node, (TypeError, ValueError)) for node in nodes):
        return "invalid_endpoint_or_request"
    names = " ".join(type(node).__name__.casefold() for node in _error_nodes(error))
    details = " ".join(str(node).casefold() for node in _error_nodes(error))
    if "proxy" in names or "proxy" in details:
        return "proxy_failure"
    if "timeout" in names or "timed out" in details:
        return "provider_timeout"
    if "ssl" in names or "tls" in details:
        return "tls_failure"
    if any(token in details for token in ("getaddrinfo", "dns", "name or service not known")):
        return "dns_failure"
    if any(token in names for token in ("connecterror", "connecttimeout")):
        return "network_connection_failure"
    if status is not None and status >= 500:
        return "provider_failure"
    return "provider_failure"


def get_http_status(error):
    response = getattr(error, "response", None)
    status = (
        getattr(error, "status_code", None)
        or getattr(response, "status_code", None)
        or getattr(error, "code", None)
    )
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def check_endpoint(timeout_seconds=5.0):
    """Check DNS/proxy/TLS/HTTP reachability without sending credentials."""
    started = time.perf_counter()
    request = urllib.request.Request(API_ROOT, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return True, "reachable_http_response", response.status, time.perf_counter() - started
    except urllib.error.HTTPError as error:
        # Any HTTP response proves the hostname, TLS, and HTTP route were reached.
        return True, "reachable_http_response", error.code, time.perf_counter() - started
    except Exception as error:
        return False, classify_network_error(error), None, time.perf_counter() - started


def run_diagnostic(
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    model_override=None,
    method="both",
    generation_only=False,
):
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model = model_override or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    proxy_configured = bool(urllib.request.getproxies())

    print(f"google-genai={importlib.metadata.version('google-genai')}")
    print(f"api_key_configured={'yes' if api_key else 'no'}")
    print(f"model={model}")
    print(f"proxy_configured={'yes' if proxy_configured else 'no'}")

    if generation_only:
        print("endpoint_preflight=skipped generation_only=yes")
    else:
        endpoint_ok, endpoint_category, http_status, endpoint_elapsed = check_endpoint()
        print(
            f"endpoint={API_ROOT} reachable={'yes' if endpoint_ok else 'no'} "
            f"category={endpoint_category} http_status={http_status or 'none'} "
            f"elapsed_seconds={endpoint_elapsed:.2f}"
        )

    if not api_key:
        print("model_test=not_run category=missing_api_key response=none")
        return 2

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    results = []
    try:
        if generation_only:
            if method == "models.generate_content":
                results.append(_test_generate_content(client, model))
            elif method == "interactions.create":
                results.append(_test_interactions(client, model, timeout_seconds))
            else:
                raise ValueError("generation-only mode requires one explicit SDK method")
        else:
            results.append(_test_model_availability(client, model))
            if method in ("both", "interactions.create"):
                results.append(_test_interactions(client, model, timeout_seconds))
            if method in ("both", "models.generate_content"):
                results.append(_test_generate_content(client, model))
    finally:
        try:
            client.close()
        except Exception:
            pass
    return 0 if all(result["category"] == "success" for result in results) else 1


def _test_model_availability(client, model):
    started = time.perf_counter()
    status = None
    category = "provider_failure"
    availability = "not_confirmed"
    auth_result = "not_confirmed"
    try:
        client.models.get(model=model)
        category = "success"
        availability = "confirmed"
        auth_result = "accepted_for_model_metadata"
    except Exception as error:
        category = classify_provider_error(error)
        status = get_http_status(error)
        if category == "authentication_failure":
            auth_result = "rejected"
        elif category == "permission_or_api_access_failure":
            auth_result = "access_denied"
        elif category == "invalid_model":
            availability = "not_available"
    print(
        f"method=models.get model={model} category={category} "
        f"http_status={status if status is not None else ('2xx' if category == 'success' else 'none')} "
        f"availability={availability} auth_result={auth_result} "
        f"elapsed_seconds={time.perf_counter() - started:.2f}"
    )
    return {"category": category}


def _test_interactions(client, model, timeout_seconds):
    started = time.perf_counter()
    status = None
    category = "provider_failure"
    response_status = "empty"
    try:
        response = client.interactions.create(
            model=model,
            input=DIAGNOSTIC_PROMPT,
            timeout=timeout_seconds,
            generation_config={"thinking_level": "low", "max_output_tokens": 64},
        )
        output = getattr(response, "output_text", None)
        if isinstance(output, str) and output.strip():
            category = "success"
            response_status = "non_empty"
    except Exception as error:
        category = classify_provider_error(error)
        status = get_http_status(error)
    result = {"category": category}
    _print_method_result("interactions.create", model, category, response_status, status,
                         time.perf_counter() - started)
    return result


def _test_generate_content(client, model):
    started = time.perf_counter()
    status = None
    category = "provider_failure"
    response_status = "empty"
    try:
        response = client.models.generate_content(
            model=model,
            contents=DIAGNOSTIC_PROMPT,
            config=types.GenerateContentConfig(
                max_output_tokens=64,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        output = getattr(response, "text", None)
        if isinstance(output, str) and output.strip():
            category = "success"
            response_status = "non_empty"
    except Exception as error:
        category = classify_provider_error(error)
        status = get_http_status(error)
    result = {"category": category}
    _print_method_result("models.generate_content", model, category, response_status, status,
                         time.perf_counter() - started)
    return result


def _print_method_result(method, model, category, response_status, status, elapsed):
    print(
        f"method={method} model={model} category={category} "
        f"http_status={status if status is not None else ('2xx' if category == 'success' else 'none')} "
        f"response={response_status} elapsed_seconds={elapsed:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-call timeout in seconds (default: 12; maximum: 15).",
    )
    parser.add_argument(
        "--model",
        help="Override the environment model for this diagnostic only.",
    )
    parser.add_argument(
        "--method",
        choices=("both", "interactions.create", "models.generate_content"),
        default="both",
        help="SDK generation method to test.",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Skip endpoint and model metadata requests; make one generation call only.",
    )
    args = parser.parse_args()
    if not 1.0 <= args.timeout <= 15.0:
        parser.error("--timeout must be between 1 and 15 seconds")
    if args.generation_only and args.method == "both":
        parser.error("--generation-only requires --method interactions.create or models.generate_content")
    return run_diagnostic(
        args.timeout,
        model_override=args.model,
        method=args.method,
        generation_only=args.generation_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())

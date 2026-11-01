import socket
import ssl
import unittest

from scripts.gemini_diagnostic import classify_network_error, classify_provider_error


class _StatusError(Exception):
    def __init__(self, status_code, message="provider response"):
        super().__init__(message)
        self.status_code = status_code


class GeminiDiagnosticTests(unittest.TestCase):
    def test_provider_statuses_are_classified_without_exposing_messages(self):
        self.assertEqual(classify_provider_error(_StatusError(401)), "authentication_failure")
        self.assertEqual(
            classify_provider_error(_StatusError(403)), "permission_or_api_access_failure"
        )
        self.assertEqual(classify_provider_error(_StatusError(429)), "rate_limit")
        self.assertEqual(
            classify_provider_error(_StatusError(404)), "invalid_endpoint_or_request"
        )
        self.assertEqual(
            classify_provider_error(_StatusError(404, "model not found")), "invalid_model"
        )
        self.assertEqual(
            classify_provider_error(_StatusError(400, "invalid request")),
            "invalid_endpoint_or_request",
        )
        self.assertEqual(classify_provider_error(_StatusError(504)), "provider_timeout")
        self.assertEqual(classify_provider_error(_StatusError(503)), "provider_failure")

    def test_network_categories_distinguish_dns_tls_connection_and_timeout(self):
        self.assertEqual(classify_network_error(socket.gaierror("private host text")), "dns_failure")
        self.assertEqual(classify_network_error(ssl.SSLError("private TLS detail")), "tls_failure")
        self.assertEqual(
            classify_network_error(ConnectionError("private connection detail")),
            "network_connection_failure",
        )
        self.assertEqual(classify_network_error(TimeoutError("private timeout detail")), "timeout")

    def test_provider_network_errors_keep_the_transport_category(self):
        self.assertEqual(classify_provider_error(socket.gaierror("private host text")), "dns_failure")
        self.assertEqual(classify_provider_error(ssl.SSLError("private TLS detail")), "tls_failure")
        self.assertEqual(
            classify_provider_error(TimeoutError("private timeout detail")), "provider_timeout"
        )


if __name__ == "__main__":
    unittest.main()

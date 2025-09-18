from loguru import logger

from neuralshield.preprocessing.pipeline import preprocess


def build_mock_http_request() -> str:
    """
    Build a raw HTTP/1.1 request with CRLF line endings.

    The request includes:
    - Duplicated query parameters and encoded values
    - Mixed-case headers
    - An example obs-fold (folded) header line
    - A duplicate header name
    """
    return (
        "GET /api/search?q=foo%252Ebar&user=John+Doe&flag&empty= HTTP/1.1\r\n"
        "Host: Example.com\r\n"
        "User-Agent: Mozilla/5.0\r\n"
        "Accept: text/html\r\n"
        "Accept: */*\r\n"
        "X-Custom: first line\r\n"
        "\tsecond\tpart continued\r\n"
        "  third part\r\n"
        "\r\n"
    )


def main():
    raw_request = build_mock_http_request()

    result = preprocess(raw_request)

    logger.info("==== RAW REQUEST (truncated preview) ====")
    logger.info(raw_request.split("\r\n\r\n")[0] + "\r\n\r\n...")
    logger.info("\n==== STRUCTURED OUTPUT ====")
    logger.info(result)


if __name__ == "__main__":
    main()

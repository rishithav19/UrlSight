import argparse
import json
from .scraper import WebScraper, UnsafeURLError

def main():
    parser = argparse.ArgumentParser(description="URLSight public webpage scraper")
    parser.add_argument("url")
    parser.add_argument("--name", default="page")
    parser.add_argument("--output", default="webscraper_output")
    parser.add_argument("--timeout", type=int, default=20000)
    args = parser.parse_args()
    try:
        result = WebScraper(args.output, args.timeout).scrape(args.url, args.name)
    except UnsafeURLError as exc:
        parser.error(str(exc))
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1

if __name__ == "__main__":
    raise SystemExit(main())

import argparse

from loguru import logger as log

from evaluation_preparation.scraping import Scraper


def main():
    parser = argparse.ArgumentParser(description="Image extraction")

    parser.add_argument("--driver_path", type=str)
    parser.add_argument("--sentences_path", type=str)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()
    scraper = Scraper(
        args.driver_path, args.sentences_path, args.output_folder
    )
    log.info("Scraping started")
    scraper()
    log.info("Scraping finished")


if __name__ == "__main__":
    main()

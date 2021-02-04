import time
import urllib.request

import numpy as np
from loguru import logger as log
from selenium import webdriver


class Scraper:
    def __init__(
        self, driver_path: str, sentences_path: str, output_folder: str
    ) -> None:
        self.driver = webdriver.Chrome(driver_path)
        self.f = open(sentences_path, "r")
        self.url = "https://images.google.com/imghp?hl=en&gl=ar&gws_rd=ssl"
        self.output_folder = output_folder

    def __call__(self):
        sentences = [" ".join(x.split()[1:-1]) for x in self.f.readlines()]

        for s in sentences:
            self.driver.get(self.url)
            d = self.driver.find_element_by_name("q")
            d.send_keys(s)
            d.submit()
            image_links = self.driver.find_elements_by_class_name(
                "rg_i.Q4LuWd"
            )
            src_links = [
                image_links[i].get_attribute("src")
                for i in range(len(image_links))
            ]

            name = f"{self.output_folder}/{'_'.join(s.split())}.jpeg"

            urllib.request.urlretrieve(src_links[0], name)
            log.info(s)
            time.sleep(np.random.choice(1))

        self.driver.quit()

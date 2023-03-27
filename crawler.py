import os
from icrawler.builtin import BingImageCrawler

from itertools import product

def crawl_bing(save_path, keyword, max_num=250):
    bing_crawler = BingImageCrawler(downloader_threads=4,
                    storage={'root_dir': save_path})
    bing_crawler.crawl(keyword=keyword, filters=None, offset=0, max_num=max_num)


# dreamstime, alamy, depositphotos image sites have white background images
if __name__ == "__main__":

    descriptors = [["lecture hall", "court", "metro", "hong kong"],
                   ["looking down", "looking up"],
                   ["steps", "stairwell"]]

    permutations = product(*descriptors)
    keywords = [" ".join(x) for x in permutations]

    for kw in keywords:
        crawl_bing(os.path.join("img_crawler","imgs","stair detection", kw), kw, max_num=30)
    

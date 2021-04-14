# usr/bin/env python
# -*- coding: UTF-8 -*-

from time import sleep
import sys
import os
import random
python_path=os.environ.get('python_path')
if python_path and python_path not in sys.path:
    sys.path.append(python_path)
import json
import logging
from page_to_vector import *
from redis import Redis
from lxml import etree
from facility import *
from ac_automation import *
from multiprocessing import Process
from multiprocessing import Queue as ProcessQueue
from crawldoc.ttypes import *
from crawl_handler import fetcher_manager
reload(sys)
sys.setdefaultencoding('utf-8')

class ResultHandler(Process):
    def __init__(self, input_queue):
        super(ResultHandler, self).__init__()
        self.input_queue = input_queue
        self.page_to_vector = PageToVec()
        self.vec_list = []
        self.id_url_list = []
        self.total_num = 0

    def run(self):
        while 1:
            doc = self.input_queue.get()
            if doc.content == None or doc.code != 200 or len(doc.content) <= 2048:
                logger.error('crawl fail: %d,%s' % (doc.code,doc.url))
                continue
            lines = doc.content.strip().split('\n')
            host = doc.url.strip().split('/')[2]
            domain = ".".join(host.split('.')[1:])
            vec = self.page_to_vector.parse(doc.url, lines, domain)
            self.page_to_vector.erase()
            if vec and len(vec) == 512:
                self.vec_list.append(vec)
                d = json.loads(doc.attachment_json)
                id_ = d["id"]
                self.id_url_list.append(id_ + "\t" + doc.url)
            if len(self.vec_list) == 200:
                with open("neg_samples.txt", "a+") as f:
                    for vec in self.vec_list:
                        s = ",".join(map(str,vec)) + ",0\n"
                        f.write(s)
                        self.total_num += 1
                        logger.info("Total num: %d" % self.total_num)
                    self.vec_list = []
                with open("neg_id_url.txt", "a+") as f:
                    for item in self.id_url_list:
                        f.write(item + "\n")
                    self.id_url_list = []

def sample():
    d = {}
    res = []
    with open('./data/goral_neg_training_data.txt', 'r') as f:
        lst = f.readlines()
        for line in lst:
            host = line.strip().split('/')[2]
            if host in d and d[host] >= 30:
                continue
            elif host in d and d[host] < 30:
                res.append(line.strip() + "\n")
                v = d[host]
                d[host] += 1
            else:
                res.append(line.strip() + "\n")
                d[host] = 1
    with open('./data/neg_training_data.txt', 'w') as f:
        f.writelines(res)

def main():
    #sample()
    input_queue = ProcessQueue(5000)
    result_handler = ResultHandler(input_queue)
    result_handler.start()
    fetcher = fetcher_manager.FetcherManager(input_queue, 200, "crawler/python/normal_fetcher/0")
    fetcher.start()
    with open('./data/neg_training_data.txt', 'r') as f:
        num = 0
        for line in f.readlines():
            doc = CrawlDoc()
            doc.url = line.strip()
            doc.request_url = line.strip()
            doc.fetch_type = 0
            doc.recrawl_times = 3
            num += 1
            js = {}
            js["id"] = str(num)
            doc.attachment_json = json.dumps(js, ensure_ascii=False)
            fetcher.put(doc)

if __name__ == '__main__':
    main()

# usr/bin/env python
# -*- coding: UTF-8 -*-

import time, datetime
from time import sleep
import sys
import os
import math
import urllib
import random
python_path=os.environ.get('python_path')
if python_path and python_path not in sys.path:
    sys.path.append(python_path)
import logging
import copy
from lxml import etree
from facility import *
from ac_automation import *
#reload(sys)
#sys.setdefaultencoding('utf-8')

class PageToVec(object):
    def __init__(self):
        super(PageToVec, self).__init__()
        self.domain = ""
        self.host = ""
        self.url = ""
        # url最长限制
        self.min_url_len = 256
        self.vector = []
        # max vector len
        self.vsize = 512
        self.h_tags = ['h1', 'h2']
        # 无需递归处理的标签
        self.skip_tags = ["footer","noscript","nav","input","select","option"]
        self.kw_dict = {}
        # 标点列表
        self.punc_list = ["。", "；", "：", "，", "？", "！"]
        # AC自动机
        self.ac_automation = ac_automation()
        self.ac_automation.parse("../data/key_word.txt")
        # 段落最小长度
        self.min_para_len = 16
        # 段落长度标准差
        self.para_len_stddev_threshold = 16
        # 段落每行平均字数
        self.ave_char_num_per_line = 32
        # 标题最小长度
        self.min_h_len = 8
        # p 标签数
        self.p_num = 0
        # a 标签数
        self.a_num = 0
        # big para num
        self.big_para_threhold = 128
        self.big_para_num = 0
        # anchor最短长度
        self.min_anchor_len = 5
        # para length vector
        self.para_len_vec = []
        # others
        self.has_video = False
        self.has_embed = False
        self.has_audio = False
        self.has_hide_media = False
        self.has_title = False
        self.has_big_img = False
        # url_anchor list
        self.url_anchor_list = []

    def erase(self):
        self.vector = []
        self.para_len_vec = []
        self.url_anchor_list = []
        self.p_num = 0
        self.a_num = 0
        self.big_para_num = 0
        self.domain = ""
        self.host = ""
        self.url = ""
        self.has_video = False
        self.has_embed = False
        self.has_audio = False
        self.has_hide_media = False
        self.has_title = False
        self.has_big_img = False

    def parse(self, url, content, domain=""):
        self.erase()
        try:
            host = url.split('/')[2]
            self.url = url
            self.host = host
            self.domain = domain
            self.clearNotes(content)
            self.clearStyle(content)
            lst = []
            for line in content:
                if line == "*\n" or line == "#\n" or line == "~\n":
                    continue
                lst.append(line)
            #with open("html.out", "a+") as f:
            #    f.writelines(lst)
            if not lst or len(lst) == 0:
                return []
            html = etree.HTML("".join(lst))
            if html is None:
                return []
            tree = etree.ElementTree(html)
            node_list = tree.xpath('/html/body')
            if node_list is not None and len(node_list) > 0:
                self.fill(node_list[0])
                self.normalize()
                return self.vector
            else:
                self.vector = []
                return self.vector
        except Exception as e:
            logger.info("error: %s" % str(e))
            return self.vector

    def is_content(self):
        has_video = False
        has_hide_media = False
        has_title = False
        has_big_para = False
        has_big_img = False
        too_many_para = False
        if len(self.vector) != self.vsize:
            return False
        if self.has_video:
            has_video = True
        if self.has_hide_media:
            has_hide_media = True
        if self.has_title:
            has_title = True
        if self.big_para_num > 0:
            has_big_para = True
        if self.has_big_img:
            has_big_img = True
        i = 0
        while i < self.vsize:
            if self.vector[i] != 1:
                i += 1
                continue
            c = 0
            while i < self.vsize and self.vector[i] == 1:
                c += 1
                i += 1
            if c >= 3:
                too_many_para = True
                break
        if has_title:
            return True
        if has_video or has_hide_media:
            return True
        if has_big_img:
            return True
        if too_many_para or has_big_para:
            return True
        if self.a_num == 0:
            return True
        return False

    def is_hub(self):
        if len(self.vector) != self.vsize:
            return False
        a_ratio = float(self.a_num) / float(self.vsize)
        if (self.has_video or self.has_embed) and a_ratio < 0.2:
            logger.info("has video or embed: %s" % self.url)
            return False
        if self.has_hide_media and a_ratio < 0.2:
            logger.info("has hide media in script: %s" % self.url)
            return False
        if self.has_big_img and a_ratio < 0.2:
            logger.info("has big image in text: %s" % self.url)
            return False
        if self.has_title and a_ratio <= 0.2:
            logger.info("has h1 tag: %s" % self.url)
            return False
        if self.big_para_num > 0 and a_ratio <= 0.2:
            logger.info("has big para: %s" % self.url)
            return False
        i = 0
        hub = True
        while i < self.vsize:
            if self.vector[i] != 1:
                i += 1
                continue
            c = 0
            while i < self.vsize and self.vector[i] == 1:
                c += 1
                i += 1
            if c >= 3:
                hub = False
                break
        if not hub:
            return False
        n = len(self.para_len_vec)
        if n == 0:
            return self.a_num >= 5
        elif n == 1 and a_ratio <= 0.2:
            return False
        else:
            if a_ratio > 0.2:
                return True
            # compute para length mean an var
            ave = sum(self.para_len_vec) / n
            var = 0
            res = True
            for e in self.para_len_vec:
                var += (e-ave) ** 2
                if e - ave >= 2 * self.para_len_stddev_threshold:
                    res = False
                    break
            stddev = math.sqrt(var/n)
            if self.a_num >= 5 and res and stddev <= self.para_len_stddev_threshold:
                logger.info("var is valid: %s" % self.url)
            return self.a_num >= 5 and res and stddev <= self.para_len_stddev_threshold

    """
    valid paragraph length > 10
    """
    def is_valid_para(self, para):
        if para == None or para == "":
            return False
        res = False
        for e in self.punc_list:
            if para.endswith(e):
                res = True
        if res:
            return True
        ln = len(para)
        if ln >= self.min_para_len:
            return True
        else:
            return False

    """
    fill vector by iterate node tree
    param: tree root
    param: recuresive depth
    """
    def fill(self, root):
        for node in root.getchildren():
            if self.skip(node):
                continue
            childs = node.getchildren()
            # a tag process
            if node.tag == "a":
                self.tag_a(node)
                continue
            # h tag process
            if node.tag in self.h_tags:
                h = self.tag_h(node)
                if h and len(h) >= self.min_h_len:
                    self.vector.append("h-" + str(len(h)))
                else:
                    self.vector.append("0")
                continue
            # img tag process
            if node and node.get("src"):
                self.tag_img(node)
                continue
            if node.tag == "script":
                self.tag_script(node)
                continue
            if node.tag == "iframe":
                self.tag_iframe(node)
                continue
            if node.tag == "video":
                self.tag_video(node)
                continue
            if node.tag == "embed":
                self.tag_embed(node)
                continue
            if node.tag == "audio":
                self.tag_audio(node)
                continue
            # br tag
            if node.tag == "br":
                self.vector.append("p-0")
                continue
            # paragragh process
            if node.tag == "p" or (not childs or len(childs) == 0):
                level = 0
                para = self.tag_p(node, level)
                if self.is_valid_para(para):
                    self.para_len_vec.append(len(para))
                    self.vector.append("p-" + str(len(para)))
                else:
                    self.vector.append("0")
                if para and len(para) >= self.big_para_threhold:
                    self.big_para_num += 1
                continue
            self.fill(node)

    """
    normalize vector so that all of the element bounds in [0,1,-1]
    h: 1
    p: 1
    a: -1
    others: 0
    """
    def normalize(self):
        ln = len(self.vector)
        self.p_num = 0
        self.a_num = 0
        if ln <= 0:
            return
        # phase one: map
        for i in range(ln-1, -1, -1):
            c = self.vector[i][0]
            # omit
            if c == '0':
                self.vector[i] = 0
                continue
            # <a>
            if c == '-':
                self.a_num += 1
                self.vector[i] = -1
                continue
            # head
            if c == 'h':
                self.vector[i] = 'h'
                continue
            # <img>
            if c == 'i':
                self.vector[i] = 'i'
                continue
            # media, such as video, embed, etc
            if c == 'm':
                self.vector[i] = 'm'
                continue
            # <p>
            line_num = max(1, int(self.vector[i].split('-')[1]) / self.ave_char_num_per_line)
            if c == 'p':
                self.p_num += 1
                self.vector[i] = line_num
            else:
                self.vector[i] = 0
        # phase two: expand
        container = []
        for e in self.vector:
            if str(e) == '-1':
                container.append(e)
            elif e == 'h':
                container.extend([1] * 10)
            elif e == 'i':
                container.extend([1] * 5)
            elif e == 'm':
                container.extend([1] * 10)
            else:
                container.extend([1] * int(e))
        #logger.info("num: %d, %d, %d, %s" % (len(self.vector), self.p_num, self.a_num, self.url))
        self.vector = container
        ln = len(self.vector)
        if ln > self.vsize:
            self.zoom()
            return
        if ln < self.vsize:
            self.vector.extend([0] * (self.vsize - ln))

    def zoom(self):
        self.zoom_internal(5, 0)
        if len(self.vector) == self.vsize:
            return
        self.zoom_internal(5, -1)
        if len(self.vector) > self.vsize:
            self.truncate()

    def zoom_internal(self, loop_num = 3, fig = 0):
        loop = 0
        while loop < loop_num:
            n = len(self.vector)
            df = n - self.vsize
            i = j = 0
            v = []
            while True:
                while i < n and self.vector[i] != fig:
                    i += 1
                v.extend(self.vector[j : i])
                if i >= n:
                    break
                j = i
                while i < n and self.vector[i] == fig:
                    i += 1
                if i - j > 1:
                    ln = 0
                    if i - j - df >= 1:
                        ln = i - j - df
                        df = 0
                    else:
                        r1 = int(random.uniform(1, i-j-1))
                        r2 = int(random.uniform(1, i-j-1))
                        r3 = int(random.uniform(1, i-j-1))
                        ln = min(min(r1,r2), r3)
                        df = df - (i - j - ln)
                    j = i
                    v.extend([fig] * ln)
                if df == 0:
                    v.extend(self.vector[i:])
                    break
                i += 1
            self.vector = v
            if df == 0:
                break
            loop += 1

    def truncate(self):
        j = len(self.vector) - self.vsize
        hzn = tzn = 0
        p = 0
        q = -1
        c = 0
        while c < j:
            hzn += int(self.vector[p] == 0)
            tzn += int(self.vector[q] == 0)
            c += 1
            p += 1
            q -= 1
        if hzn > tzn:
            self.vector = self.vector[j:]
        else:
            self.vector = self.vector[0 : self.vsize]

    """
    skip redundant nodes, such as attributes containing nav, ad, footer, etc
    root: current node
    """
    def skip(self, root):
        if root.tag in self.skip_tags:
            return True
        d = root.items()
        for item in d:
            if item[0] != "id" and item[0] != "class":
                continue
            if "video" in item[1] or "embed" in item[1] or "audio" in item[1]:
                continue
            if self.ac_automation.hit_key_words(item[1]):
                #logger.info("attr value: %s" % item[1])
                return True
        return False

    """
    filter key words by AC-Automation
    text: filted text
    """
    def hit_key_word(self, text):
        if not text or not text.strip():
            return False
        b1 = False
        b2 = False
        if len(text.replace("None","").strip()) < self.min_para_len:
            b1 = True
        else:
            return False
        if self.ac_automation.hit_key_words(text.strip().encode("utf-8")):
            b2 = True
        return b1 and b2

    """
    process a tag
    node: current node
    """
    def tag_a(self, node):
        href = node.get("href")
        if not href:
            self.vector.append('0')
            return
        # maybe advertisement
        if len(href) > self.min_url_len:
            return

        t = node
        while True:
            if t.text and t.text.strip():
                break
            c = t.getchildren()
            if not c or len(c) == 0:
                break
            t = c[0]
            nd = None
            for sc in c:
                if sc.text:
                    nd = sc
                    t = nd
                    break
        self.url_anchor_list.append((href, t.text))

        # hit key word
        if self.hit_key_word(t.text):
            self.vector.append('0')
            return
        # anchor too short
        if t.text and len(t.text) < self.min_anchor_len:
            self.vector.append('0')
            return
        # such as /a/b.html, ../a.html, etc
        if not href.startswith("http") and "/" in href:
            self.vector.append('-1')
            return
        self.vector.append('-1')
        '''
        if self.domain and self.domain in href:
            self.vector.append('-1')
        else:
            self.vector.append('0')
        '''

    def getList(self):
        return self.url_anchor_list

    '''
    process p node and all sub nodes of p recrusivly
    param: current node
    param: record depth
    '''
    def tag_p(self, node, level):
        text = ""
        if node.text:
            text += node.text
        if node.tail:
            text += node.tail
        # get current text of node
        if text and not self.hit_key_word(text):
            text = text.strip()
        else:
            text = ""
        # get all text of node's subnodes recrusivly
        childs = node.getchildren()
        if childs and len(childs) > 0:
            level += 1
        for c in childs:
            if c.tag == 'a':
                self.tag_a(c)
                continue
            if c.tag in self.h_tags:
                self.tag_h(c)
                continue
            if c.tag == "br" and text and len(text) >= self.min_para_len:
                self.vector.append("p-0")
                continue
            if c.tag == "iframe":
                self.tag_iframe(c)
                continue
            if c and c.tag == "img" and level <= 2 and c.get("src"):
                self.tag_img(c)
                continue
            text += self.tag_p(c, level)
        return text.replace("None", "")

    def tag_img(self, node):
        if ".jpg" not in node.get("src") and ".png" not in node.get("src"):
            return
        p = node.getparent()
        if not p:
            return
        if p.tag != "p" and p.tag != "span" and p.tag != "center" and p.tag != "div":
            return
        h = node.get("height")
        w = node.get("width")
        if (h and int(h) < 180) or (w and int(w) < 180):
            return
        self.vector.append("img")
        self.has_big_img = True

    '''
    process h node and all sub nodes of node
    param node: current h node
    '''
    def tag_h(self, node):
        sentry = False
        text = ""
        if node.text:
            text += node.text
        if node.tail:
            text += node.tail
        # get current text of node
        if text and not self.hit_key_word(text) and len(text.strip()) >= self.min_h_len:
            text = text.strip()
        else:
            text = ""
        if text and len(text) > self.min_h_len and not self.hit_key_word(text):
            self.has_title = True
            sentry = True
        # get all text of node's subnodes recrusivly
        childs = node.getchildren()
        for c in childs:
            if c.tag == 'a':
                if sentry and c.text and c.text.find(text) != -1:
                    self.has_title = False
                self.tag_a(c)
                continue
            text += self.tag_h(c)
        return text.replace("None", "")

    '''
        detect media hiding in script
    '''
    def tag_script(self, node):
        if self.has_video or self.has_audio or self.has_embed:
            return
        src = node.get("src")
        if src and src.find(".mp4") != -1:
            self.has_video = True
            return
        if src and src.find(".mp3") != -1:
            self.has_audio = True
            return
        score = 0
        text = node.text
        if not text:
            return
        token = ""
        if text.find(".mp4") != -1:
            score += 4
            token += ".mp4_"
        if text.find(".mp3") != -1:
            score += 4
            token += ".mp3_"
        if text.find("<video") != -1:
            score += 3
            token += "<video_"
        if text.find("Player") != -1 or text.find("player") != -1:
            score += 2
            token += "Player_"
        if text.find("width") != -1 or text.find("height") != -1:
            score += 1
            token += "wh_"
        if text.find("Width") != -1 or text.find("Height") != -1:
            score += 1
            token += "wh_"
        if text.find("video") != -1 or text.find("Video") != -1:
            score += 1
            token += "video_"
        if score >= 5:
            logger.info("score >=5: %s, %s" % (self.url, token))
            self.vector.append("media")
            self.has_hide_media = True

    def tag_iframe(self, node):
        if self.has_video or self.has_audio or self.has_embed:
            return
        keys = node.keys()
        if "src" in keys and "width" in keys and "height" in keys:
            src = node.get("src")
            if src.find(".mp4") != -1:
                self.has_video = True
                return
            if src.find(".mp3") != -1:
                self.has_audio = True
                return
            if src.find("video") != -1 or src.find("Video") != -1:
                self.has_hide_media = True
                return
            if src.find("player") != -1 or src.find("Player") != -1:
                self.has_hide_media = True
                return
        childs = node.getchildren()
        for node in childs:
            if node.tag == "script":
                self.tag_script(node)
            elif node.tag == "video":
                self.tag_video(node)
            elif node.tag == "audio":
                self.tag_audio(node)
            else:
                continue

    def tag_video(self, node):
        self.has_video = True
        self.vector.append("media")

    def tag_embed(self, node):
        self.has_embed = True
        self.vector.append("media")

    def tag_audio(self, node):
        self.has_audio = True
        self.vector.append("media")

    def clearScripts(self, content):
        n = len(content)
        i = 0
        while i < n:
            if not content[i].strip().startswith("<script"):
                i += 1
                continue
            while i < n and not content[i].strip().endswith("</script>"):
                content[i] = "#\n"
                i += 1
            if i < n:
                content[i] = "#\n"
                i += 1

    def clearStyle(self, content):
        n = len(content)
        i = 0
        while i < n:
            si = content[i].find("<style")
            if si == -1:
                i += 1
                continue
            ei = -1
            enjambment = False
            while i < n:
                ei = content[i].find("</style>")
                if ei == -1:
                    enjambment = True
                    content[i] = "~\n"
                    i += 1
                    continue
                if not enjambment:
                    splitor = content[i][si:ei+8]
                    content[i] = "".join(content[i].split(splitor))
                else:
                    content[i] = content[i][ei+8:]
                break

    def clearNotes(self, content):
        n = len(content)
        i = 0
        while i < n:
            si = content[i].find("<!--")
            if si == -1:
                i += 1
                continue
            ei = -1
            enjambment = False
            while i < n:
                ei = content[i].find("-->")
                if ei == -1:
                    enjambment = True
                    content[i] = "*\n"
                    i += 1
                    continue
                if not enjambment:
                    splitor = content[i][si:ei+3]
                    content[i] = "".join(content[i].split(splitor))
                else:
                    content[i] = content[i][ei+3:]
                break

if __name__ == '__main__':
    url = "http://www.fs024.com/video/video-52141.html"
    url = "http://www.zx.chengdu.gov.cn/"
    url = "http://orth.cmt.com.cn"
    url = "http://xy.cnhubei.com/xyjjjr/p/12806455.html"
    domain = "cnhubei.com"
    lines = []
    with open("a.txt", "r") as f:
        for line in f.readlines():
            lines.append(line)
    page_to_vec = PageToVec()
    page_to_vec.parse(url, lines, domain)
    print(page_to_vec.vector)
    print(page_to_vec.has_title)
    print(page_to_vec.has_video)
    print(page_to_vec.has_audio)
    print(page_to_vec.has_embed)
    print(page_to_vec.has_big_img)
    print(page_to_vec.big_para_num)
    print(page_to_vec.a_num)
    print(page_to_vec.has_hide_media)
    print(page_to_vec.is_hub())

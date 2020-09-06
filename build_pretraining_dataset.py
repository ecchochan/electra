# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes out text data as tfrecords that ELECTRA can be pre-trained on."""

import argparse
import multiprocessing
import os
import random
import time
import tensorflow.compat.v1 as tf

from model import tokenization
from util import utils

from cantokenizer import CanTokenizer
from replacer import Replacer
import re
chinese_re = re.compile(u' *([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]) *', re.UNICODE)

seen = set()

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class ExampleBuilder(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, tokenizer, max_length, do_sop=False, do_cluster=False):
    self._tokenizer = tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length if not do_cluster else max_length * 2
    self.do_sop = do_sop
    self.do_cluster = do_cluster
    self.warned = False

  def add_line(self, line, input_file=None):
    """Adds a line of text to the current example being built."""
    line = line.strip().replace("\n", " ")
    if (not line) and self._current_length != 0:  # empty lines separate docs
      return self._create_example()
    encoded = self._tokenizer.encode(line)
    # bert_tokens = encoded.tokens
    bert_tokids = encoded.ids 

    unk_count = bert_tokids.count(4)
    
    '''
    if unk_count > 0:
      p = bert_tokids.index(4)
      offsets = encoded.offsets
      tokenized_text = ' '.join(encoded.tokens[p-10:p+10])
      if offsets[p][1] - offsets[p][0] == 1:
        unk_token_should_be = line[offsets[p][0]:offsets[p][1]]
        if unk_token_should_be not in seen:
          seen.add(unk_token_should_be)
          orig_text = line[offsets[p][0]-10:offsets[p][1]+10]
          tokenized_text = chinese_re.sub(r'\1',tokenized_text)
          print(tokenized_text+'\n'+ orig_text)

    if random.random() < 0.0001:
      tokenized_text = ' '.join(encoded.tokens[:40])
      orig_text = line[:40]
      tokenized_text = chinese_re.sub(r'\1',tokenized_text)
      print(input_file +'\n'+tokenized_text+'\n'+ orig_text)


'''
    if unk_count > 15:
      return self._create_example()


    self._current_sentences.append(bert_tokids)
    self._current_length += len(bert_tokids)
    if self._current_length >= self._target_length:
      if self.do_cluster and len(self._current_sentences) <= 1:
        return None
      return self._create_example()
    return None

  def make_segments(self, sentences):
    if not self.do_sop and random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      ss = (self._max_length - 3) // 2
      first_segment_target_length = ss if not self.do_sop else \
                                    (random.randint(min(8,ss), ss) if random.random() > 0.5 else ss
                                    )

    first_segment = []
    second_segment = []
    if self.do_sop and len(sentences) == 1:
      length = len(sentences[0])
      a = sentences[0][:length // 2]
      b = sentences[0][length // 2:]
      sentences = [a, b]

    sep = random.randint(1,len(sentences) - 1)
    first_segment = sentences[:sep]
    second_segment = sentences[sep:]

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    if self.do_sop:
      min_seg_length = random.randint(8, 32)
      first_max_length = (self._max_length - 3 - min_seg_length)             # 256 - 3 - 32 = 221
      first_segment = first_segment[max(0,                                   # 
                                    len(first_segment) - first_max_length):] # len(segment) = 300
                                                                            # -> 300- 221 = 79
                                                                            # -> [79:] 
                                                                            # 
                                                                            # len(segment) = 64
                                                                            # -> 64 - 221 = 0
                                                                            # -> [0:] 
                                                                            # 

      second_max_length = self._max_length - 3 - len(first_segment)          # 256 - 3 - 221 = 32
      second_segment = second_segment[:second_max_length]
    else:
      first_segment = first_segment[:self._max_length - 2]
      second_segment = second_segment[:max(0, self._max_length -
                                          len(first_segment) - 3)]


    sop = None
    if self.do_sop:
      sop = 1 
      if random.random() > 0.5:
        temp = first_segment
        first_segment = second_segment
        second_segment = temp
        sop = 0

    return first_segment, second_segment, sop


  def _create_example(self):
    if not self._current_sentences:
      return None
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if not self.warned and self.do_sop:
      print("Creating tfrecords with SOP objective.")
      self.warned = True
      

    if self.do_cluster:
      num_sentences = len(self._current_sentences)
      if num_sentences == 1:
        return None


      sep = num_sentences // 2 if random.random() > 0.5 else random.randint(1, num_sentences - 1)

      A_sentences = self._current_sentences[:sep]
      B_sentences = self._current_sentences[sep:]
      A_first_segment, A_second_segment, A_sop = self.make_segments(A_sentences)
      B_first_segment, B_second_segment, B_sop = self.make_segments(B_sentences)
      A_feature = self._make_tf_example(A_first_segment, A_second_segment, A_sop, return_feature=True)
      B_feature = self._make_tf_example(B_first_segment, B_second_segment, B_sop, return_feature=True)

      for k, v in B_feature.items():
        A_feature[k+'2'] = v
      ret = tf.train.Example(features=tf.train.Features(feature=A_feature))

    else:
      first_segment, second_segment, sop = self.make_segments(self._current_sentences)
    
      ret = self._make_tf_example(first_segment, second_segment, sop)


    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0

    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length if not self.do_cluster else max_length * 2)
    else:
      self._target_length = self._max_length if not self.do_cluster else max_length * 2

    return ret

  def _make_tf_example(self, first_segment, second_segment, sop_label=None, return_feature=False):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [0]
    input_ids.extend(first_segment)
    input_ids.append(1)
    #input_ids = [0] + first_segment + [1]
    
    segment_ids = [0] * len(input_ids)
    if second_segment:
      input_ids.extend(second_segment)
      input_ids.append(1)
      segment_ids += [1] * (len(second_segment) + 1)
    input_mask = [1] * len(input_ids)
    input_ids += [0] * (self._max_length - len(input_ids))
    input_mask += [0] * (self._max_length - len(input_mask))
    segment_ids += [0] * (self._max_length - len(segment_ids))
    feature = {
        "input_ids": create_int_feature(input_ids),
        "input_mask": create_int_feature(input_mask),
        "segment_ids": create_int_feature(segment_ids)
    }
    if sop_label is not None:
      feature["sop_label"] = create_int_feature([sop_label])

    if return_feature:
      return feature

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


import re
from data_utils import too_many_repeat
remove_url_re = re.compile(r' ?(?:https?:\/\/[a-zA-Z0-9\-]+(?:\.[a-zA-Z_0-9\-]+)+|[a-zA-Z_0-9\-]+(?:\.[a-zA-Z_0-9\-]+)+)(?:\/(?:\?(?:<nl>)?\n *[a-zA-Z0-9\-\._… &%\+]+|[a-zA-Z0-9\.\?\:@\-_=#…&%!\+])+)+ *(?:<nl>\n * ?(?:https?:\/\/[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+|[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+)(?:\/(?:\?(?:<nl>)?\n *[a-zA-Z0-9\-\._… &%\+]+|[a-zA-Z0-9\.\?\:@\-_=#…&%!\+])+)+ *)*')
remove_speakers = re.compile(r'^#\d+ ([A-Z]+: )?(#\d+ )?', re.MULTILINE)

bad_unicode = re.compile(r'[\u2060-\u20ff\uAA80-\uFB45\u00AD\u008D\u008F\u009F\u0095\u0094\u0097\u0082\u0083\u0087\u0099囀ਾ]+|矛受畩究悍妤|脖宋鬱駜|ÐÒøÓÐÕ|ㄛ筍|ㄛ婌|ㄛ紨|ㄛ嘟|大虯李李朽|獝獞獟獠|拇謂饢|海瑉|隳哪|堶悸漲Л釔|野怛儞也|鈭鲭|韏啣|蟡㏘|乯儜|牁轎煤|蕻淕|蜁巌|潝砩|坉洩|竷h|匾哺|衷讜|愣勾|划曻|a﹐a¶#|p"0∨"q"|鼐盀|阠鼐|皇瞧|鍩挂槐|肭資指娓|蟛青|眩謖|笥|饇|櫱|肭|亂桓|嫠|芔苙苾苹|攆擺|似饲|恕刷|膘', 
                      re.UNICODE)
remove_borders = re.compile(r'[┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╵╷╹╻╼╽╾╿▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟◣◥͜͡╮╯╰◜◝◞◟ᕕᕗ⌇⧸⎩⎠⎞͏⎛͏⎝⎭⧹༼༽♢◄ƪʅʋ)]')
    

class ExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, job_id, vocab_file, output_dir, max_seq_length,
               num_jobs, blanks_separate_docs, do_lower_case, do_sop,
               num_out_files=1000):
    self._blanks_separate_docs = blanks_separate_docs
    tokenizer = CanTokenizer(vocab_file)
    replacements = {}
    

    simple_replacements = '''
支支梧梧,支支吾吾
愴悴,倉卒
成訧,成就
荓o天獨厚,得天獨厚
毖後,慎後
訾議,指責
無毖,無操
佢迚,佢
胗金,診金
擻旦,撒旦
擻亞,撒亞
擻手,撒手
擻下,撒下
擻冷,撒冷
看診,看診
倒禢,倒塌
然緮,然後
溘然,忽然
好胗,好彩
溘逝,忽逝
詆訿,詆毀
感䁷,感覺
胗落,胗睇
㥥然,偶然
門㰖,門檻
無䫋,無類
顪色,顏色
討讑,討論
礀勢,姿勢
暴畛,暴殄
不恂情面,不徇情面
𤓓紙,攞紙
誄程,課程
䟓爆,踢爆
上誄,上課
開誄,開課
笄芯,開心
嗫嗫嚅嚅,吞吞吐吐
而冡,而家
多蒯,多部
靚唔挸,靚唔靚
蹝左,徙左
旵士,巴士
哂萠,哂崩
穿萠,穿崩
裸 裎,赤裸
裸裎,赤裸
縻爛,糜爛
禳灾,除灾
足以⊥,足以令
濕頄,濕九
朊髒,骯髒
放簜,放盪
絾績,成績
班乹,班戟
可𠰴可以,可唔可以
校䏜,校服
凵家剷,冚家剷
雕屋脊,雕屋脊
傖促,倉促
窩錀,窩輪
資枓,資料
譊唔譊,曉唔曉
直裰,直袍
栱桿,槓桿
難堐 ,難捱
入饔,入甕
萯碟,副碟
鈠鍊,鍛鍊
龜柃,龜柃
朓望,眺望
味䁷,味覺
寒傖,寒酸
巴玎,巴打
臉脥,臉頰
譂讓,禪讓
扶筇,扶杖
躐等,越等
門襤,門檻
顝顱,骷顱
脥下,腋下
解瓿圖,解剖圖
藍笌,藍牙
枌末,粉末
止瘑,止屙
肚瘑,肚屙
治瘵,治療
闥入,闖入
鈙事,敘事
陀怫,陀佛
㩂個女人,揀個女人
無儔,無比
踽僂,傴僂
咳欶,咳嗽
䅻身,痴身
惝恍,失意
惝怳,失意
賷發,打發
睇踾到,睇唔到
齎發,打發
㧥親,跣親
㧥到,跣到
挍調,較調
㧥倒,跣倒
好㧥,好㧥
候僎,候選
影呴,影響
杪近,抄近
大棑,大排
材枓,材料
橡筯,橡筋
左猺,左搖
激璗,激盪
籟欶,籟簌
計筫,計算
動璗,動盪
熘彈,榴彈
拉筯,拉筋
推廌,推薦
得闃,得閒
𥅽好d,貪好d
接蠋,接觸
蠟蠋,蠟觸
咪菚,咪盞
旗壏,旗艦
大陰蠋,大陰燭
問顗,問題
亂椗,亂掟
懷愐,懷緬
愐懷,緬懷
十㰩,十蚊
撙抵,蹲低
咪菚,咪盞
趷低,踎低
挍一餐,拗一餐
獎過狟0,獎過期幾
陰𧖣,陰蠟
嘲文,潮文
蹝氣,嘥氣
屣氣,嘥氣
蝦誁,蝦餅
栚票,機票
齟齬,齒不齊
齟晤,齒不齊
龃龉,齒不齊
二妁,二奶
一鑤,一鑊
贂,睇
輞早,早
囁嚅,吞吐
囁嚅,吞吐
黠樣,點樣
𪘲,依
反餔,反哺
一擨,一蹶
氀好,唔好
影譠,影壇
震囁,震攝
風杋,風帆
夜䦨,夜闌
論譠,論壇
㟨士,瑞士
縯紛,繽紛
點䓹,點養
內哄,內訌
共鈙 ,共聚
唔度賩,唔到
剸入法,輸入法
Ⴡ,o靚
樂譠,影壇
海絴,海洋
追誴,追蹤
告謦,售罄
洧息,消息
三妁,三奶
o徙氣,嘥氣
沉斁,沉默
越殂,越俎
腌尖,奄尖
腌臢,奄尖
奄臢,奄尖
乏善可鯔,乏善可陳
沉𤢕,沉默
多斁,多數
多𤢕,多數
著斁,著數
著𤢕,著數
徙氣,嘥氣
落㮞,落格
造廔,造口
奔弛,奔馳
做弭,做餌
魚弭,魚餌
又泅,又lup
塵蹣,麈蟎
泅泅,lup lup
肉𤎖,肉糠
只系,只係
扭釱,扭軚
不慬,不懂
蘔果,蘋果
唔洎,唔洗
痀瘻,駝背
佝瘻,駝背
傴瘻,駝背
痀僂,駝背
佝僂,駝背
傴僂,駝背
瘳瘳,寥寥
賌料,資料
挍碎,攪碎
挍盡,攪盡
挍挍,搞搞
學挍,學校
挍晒,R曬
挍左柴,拗左柴
挍位,鉸位
挍低,較低
挍內,校內
援挍,援交
挍長,校長
挍服,校服
挍痕,R痕
有得挍,有得校
唔洗挍,唔洗拗
有排挍,有排拗
"挍"到隻,拗到隻
挍部機,較部機
挍番,較番
挍左d老,R左d老
挍西班牙呀,搞西班牙呀
"挍"得贏,拗得贏
魚挍,魚餃
同我挍,同我拗
同鼻挍,同鼻拗
挍到個,較到個
挍撚晒,R撚晒
腰挍到,拗挍到
挍柴,拗柴
挍屎棍,攪屎棍
上挍,上校
詅住,諗住
遲墿,遲擇
挍返個,較返個
就挍到,就搞到
挍緊手瓜,拗緊手瓜
唔好挍院,唔好搞院
挍生挍死,拗生拗死
挍殺王,搞殺王
勿挍事,勿搞事
挍手瓜,拗手瓜
喺度挍,喺度拗
挍痛,絞痛
挍贏,拗贏
都無數挍,都無數拗
挍到獨立,搞到獨立
挍左另外,搞左另外
死忍唔挍得,死忍唔R得
條命挍飛,條命教飛
紅波挍黑,紅波搞黑
挍極到最,較極到最
做唔成你挍,做唔成你拗
唔知你挍乜,唔知你拗乜
名挍,名校
挍剪,鉸剪
挍到死腳,拗到死腳
挍腰,拗腰
互相挍殺,互相殘殺
挍咩條仆街,搞咩條仆街
挍腳走人,拗腳走人
挍水吹,r水吹
挍返正,較返正
鄭挍唔掂,鄭搞唔掂
比佢挍到分分鐘,比佢搞到分分鐘
粗嘅就挍,粗嘅就抆
挍變速鐵鎖,較變速鐵鎖
挍斷,拗斷
挍肉機,搞肉機
挍爛,r爛
挍緊啲咩,搞緊啲咩
挍緊屎,屙緊屎
挍到佢地正常,拗到佢地正常
赢場挍,赢場交
以為唔小心挍左,以為唔小心拗左
手挍腳挍,手r腳r
慇速,光速
檢椌,檢控
申諘,申請
目摽,目標
苓膏,柃膏
猜𣐀,猜枚
猌,就
淘汱,淘汰
汰弱留,汰弱留
𨳒你焛,𨳒你𨳒
雖焛,雖然
俾人焛,俾人小
焛你,小你
焛到爆,小到爆
焛下焛下,閃下閃下
好焛,好閃
焛你,小你
邅自,擅自
焛撚晒,閃撚晒
'''

    for line in simple_replacements.split('\n'):
        if not line:
            continue
        a, b = line.split(',')
        replacements[a] = (b,'','')
    with open('combine_zh.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split(',')
            replacements[k] = (v,'','')
            
    with open('emoji_to_name.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            splitted = line.split(',')
            k, v = splitted[0][0], splitted[1]
            vocab_id = tokenizer.token_to_id(k)
            if not vocab_id:
                replacements[k] = (':%s:'%v,'','')
    self.replacer = Replacer(replacements)
    self._example_builder = ExampleBuilder(tokenizer, max_seq_length, do_sop=do_sop)
    self._writers = []
    for i in range(num_out_files):
      if i % num_jobs == job_id:
        output_fname = os.path.join(
            output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                i, num_out_files))
        self._writers.append(tf.io.TFRecordWriter(output_fname))
    self.n_written = 0

  def remove_url(self, x):
      x = remove_url_re.sub(' [url] ',x)
      x = remove_speakers.sub('',x)
      x = remove_borders.sub('',x)
      x = self.replacer.translate(x)

      return x
  def write_examples(self, input_file):
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      cached = []
      bucket = []
      for line in f:
        line = line.strip()
        if line:
          bucket.append(line)
        else:
          bucket.append("")
          sub_doc = '\n'.join(bucket)
          sub_doc = self.remove_url(sub_doc)
          bad = False
          if not sub_doc.strip() or too_many_repeat(sub_doc):
            bad = True
          if not bad:
            bad = any(1 for e in bad_unicode.finditer(sub_doc))
            
          if not bad:
            for e in re.finditer(r'[\da-zA-Z_]{10,}', sub_doc):
              g = e.group()
              if re.search(r'\d', g) and re.search(r'[A-Z]{3,}', g):
                  pass
              elif re.search(r'[a-z]+[A-Z]{2,}[a-z]+[A-Z]+', g):
                  pass
              else:
                  continue
              bad = True
              break
          if not bad:
            cached.append(sub_doc.split('\n'))
          bucket = []
      if bucket:
        bucket.append("")
        sub_doc = '\n'.join(bucket)
        sub_doc = self.remove_url(sub_doc)
        bad = False
        if not sub_doc.strip() or too_many_repeat(sub_doc):
          bad = True
        if not bad:
          for e in re.finditer(r'[\da-zA-Z_]{10,}', sub_doc):
            g = e.group()
            if re.search(r'\d', g) and re.search(r'[A-Z]{3,}', g):
                pass
            elif re.search(r'[a-z]+[A-Z]{2,}[a-z]+[A-Z]+', g):
                pass
            else:
                continue
            bad = True
            break
        if not bad:
          cached.append(bucket)

      for bucket in cached:
        for line in bucket:
          if line or self._blanks_separate_docs:
            example = self._example_builder.add_line(line, input_file)
            if example:
              self._writers[self.n_written % len(self._writers)].write(
                  example.SerializeToString())
              self.n_written += 1
      example = self._example_builder.add_line("")
      if example:
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

  def finish(self):
    for writer in self._writers:
      writer.close()


def write_examples(job_id, args):
  """A single process creating and writing out pre-processed examples."""
  def log(*args):
    msg = " ".join(map(str, args))
    print("Job {}:".format(job_id), msg)

  log("Creating example writer")
  example_writer = ExampleWriter(
      job_id=job_id,
      vocab_file=args.vocab_file,
      output_dir=args.output_dir,
      max_seq_length=args.max_seq_length,
      num_jobs=args.num_processes,
      blanks_separate_docs=args.blanks_separate_docs,
      do_lower_case=args.do_lower_case,
      do_sop=args.do_sop
  )
  log("Writing tf examples")
  fnames = sorted(tf.io.gfile.listdir(args.corpus_dir))
  fnames = [f for (i, f) in enumerate(fnames)
            if i % args.num_processes == job_id]
  random.shuffle(fnames)
  start_time = time.time()
  for file_no, fname in enumerate(fnames):
    if file_no > 0:
      elapsed = time.time() - start_time
      log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
          "{:} examples written".format(
              file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
              int((len(fnames) - file_no) / (file_no / elapsed)),
              example_writer.n_written))
    example_writer.write_examples(os.path.join(args.corpus_dir, fname))
  example_writer.finish()
  log("Done!")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--vocab-file", required=True,
                      help="Location of vocabulary file.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")
  parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                      help="Whether blank lines indicate document boundaries.")
  parser.add_argument("--do-sop", dest='do_sop',
                      action='store_true', help="Add SOP features.")
  parser.add_argument("--do-cluster", dest='do_cluster',
                      action='store_true', help="Add Cluster features.")
  parser.add_argument("--do-lower-case", dest='do_lower_case',
                      action='store_true', help="Lower case input text.")
  parser.add_argument("--no-lower-case", dest='do_lower_case',
                      action='store_false', help="Don't lower case input text.")
  parser.set_defaults(do_lower_case=True)
  args = parser.parse_args()

  utils.rmkdir(args.output_dir)
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__ == "__main__":
  main()

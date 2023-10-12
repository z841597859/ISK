import re
import spacy
from nltk import sent_tokenize
from ascent_openie import oie_from_spacy_sent

nlp = spacy.load("en_core_web_md")

index_to_reason=['',#False
                'len_of_str>200',#1
                '出现少于15次英文字母',#2
                '感叹句和问句',#3
                '性别三人称',#4
                '句首指示代词_i_me',#5
                '一二人称共现',#6
                'Include_Bad_Signals',#7
                'len_of_tok<=3or>=40',#8
                'NE-BW-pos-propn',#9
                'NE-BW-ent_type_',#10
                'NE-BW-tag-NFPADD',#11
                'NE-BW-dep-ROOT',#12
                'NE-BW-N0NBef-they',#13
                '整个句子提取不出一组spo',#14
                '所有spo都缺少主语和宾语']#15

reasons_len = len(index_to_reason)

def text_nltk_sents(text):#分句

    sents = re.sub(r'\.', '. ', text)
    sents = re.sub(r'\s+', ' ', sents)
    sents = sent_tokenize(sents)

    return sents

def Include_Bad_Signals(sent):

    if re.search('\d{4}|install|one day|the image|picture|figure|following|for free|for sale|click|[^\w]ask|'
                 '^here[^\w]|[^\w]here[^\w]|[^\w]here$|'
                 '\. \. \.|--|- -|(–|\..|\\|&|/){2,}|(\"|\'|“|’){3,}|\(|\)', sent, flags=re.IGNORECASE):# ---

        return True

    return False

def Is_Sent_Filted_1(sent):  # re \s+ ,re.sub('\(.*?\)', '', sent), lstrip() already

    len_of_str = len(sent)

    if len_of_str > 200:  # 筛掉经过nltk分句后仍然很长的句子

        return 1

    if re.search('([A-Za-z].*){15,}', sent) == None:  # 筛掉出现少于15次英文字母

        return 2

    if '!' in sent or '?' in sent:  # 筛掉经过nltk分句后的感叹句和问句

        return 3

    if re.search('[^\w]his[^\w]|[^\w]her[^\w]|[^\w]she[^\w]|[^\w]he[^\w]|[^\w]him[^\w]|[^\w]hers[^\w]|'
                 '^he[^\w]|^she[^\w]|^her[^\w]|^his[^\w]|^hers[^\w]|'
                 '[^\w]he$|[^\w]she$|[^\w]hers$|[^\w]his$|[^\w]him$|[^\w]her$', sent, flags=re.IGNORECASE):  # 筛掉出现性别第三人称

        return 4

    if re.search('^these[^\w]|^those[^\w]|^that[^\w]|^this[^\w]|^they[^\w]|'
                 '^i[^\w]|^me[^\w]|'
                 '[^\w]i[^\w]|[^\w]me[^\w]|'
                 '[^\w]me$|[^\w]i$', sent, flags=re.IGNORECASE):

        return 5

    if re.search('[^\w]we[^\w]|[^\w]our[^\w]|[^\w]us[^\w]|[^\w]ours[^\w]|[^\w]my[^\w]|'
                 '^we[^\w]|^our[^\w]|^us[^\w]^ours[^\w]|^my[^\w]|'
                 '[^\w]we$|[^\w]our$|[^\w]us$|[^\w]ours$|[^\w]my$', sent, flags=re.IGNORECASE) != None and \
            re.search('[^\w]you[^\w]|[^\w]your[^\w]|[^\w]yours[^\w]|'
                      '^you[^\w]|^your[^\w]|^yours[^\w]|'
                      '[^\w]you$|[^\w]yours$|[^\w]your$|'
                      'please', sent, flags=re.IGNORECASE) != None: #筛掉一人称二人称共现

        return 6

    if Include_Bad_Signals(sent):

        return 7

    return False

########################
# def EVI_Cut(tok):

#     if tok.lemma_ in ['argue','note','emphasises','know','reflect','believe','observe',
#                       'think','imply','feel','prove','show','found','discover','fact']:
#         return True

#     return False

# def Split_Sent(sent):

#     spans = re.split('[^\w]especially[^\w]|,but |,and |,such as |, but |, and |, such as |, then |,then | and then |'
#                       ' - | – |[^w]say[^w]]|^w]said[^w]|^w]says[^w]|[^w]because[^w]|[^w]since[^w]|'
#                       ',as |, as |as well as|[^w]so that ',sent, 1, flags=re.IGNORECASE)

#     if len(spans) == 1:

#         result = re.search('(^as [^a]|since ).+,', sent, flags=re.IGNORECASE)

#         if result != None:

#             spans = sent.split(',', 1)

#     # This is what makes me feel that big business is bound to be behind it.

#     return spans

# def Is_Span_Filted(span):
#
#     if
#
#     return False
#
# def Is_Span_Filted(span):
#
#     if
#
#     return False

########################

def Include_NE_BW(doc):

    IS_NOUN_BEFORE = 0

    for tok in doc:

        if tok.pos_ in ['PROPN','SYM','X','INTJ'] and tok.text.lower() != 'please':# PROPN 同时也会排除一些专业性很强的词汇

            return 9

        if tok.ent_type_ in ['PERSON', 'ORG', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                             'LAW', 'MONEY', 'NORP','TIME','WORK_OF_ART']:
            # 'CARDINAL','DATE','ORDINAL','PRODUCT','QUANTITY','LOC'(earth),'PERCENT'

            return 10

        if tok.tag_ in['-LRB','-RRB','NFP','ADD','$','XX','LS','FW','``',"''",':'] and tok.text!=':':#'WP','UH','IN','PDT','WP$','``',"''",':'

            return 11

        if tok.dep_ == 'meta' or (tok.dep_ == 'ROOT' and tok.tag_ == 'VBD') :

            return 12

        if tok.pos_ in ['NOUN','PRON']:

            IS_NOUN_BEFORE = 1

        if tok.text.lower() in ['they','these','those','theirs'] and IS_NOUN_BEFORE==0:

            return 13

    return False

def Is_Sent_Filted_3(sent):

    doc = nlp(sent)

    len_of_tok = len(doc)

    if len_of_tok <= 3 or len_of_tok >= 40:

        return 8

    spacy_result = Include_NE_BW(doc)

    if spacy_result != False:

        return spacy_result

    oie_result = oie_from_spacy_sent(doc)  # , get_appos=True

    if len(oie_result) == 0:  # 整个句子提取不出一组spo

        return 14

    mark = 0

    for i, a in enumerate(oie_result):

        if mark == 1:

            break

        else:

            if a['subject'] not in ['', None] or a['object'] not in ['', None]:  # 有一个spo有s或o则保留

                mark = 1

    if mark == 0:

        return 15

    return False

def main_filter(sent):

    result = Is_Sent_Filted_1(sent)

    if result != False:

        return result

    result = Is_Sent_Filted_3(sent)

    if result != False:

        return result

    return result

def main_filter_text_dict(text_dict):

    dict = {}
    dict['text'] = text_dict['text']
    dict['pass'] = []
    dict['filted'] = []

    sents = text_nltk_sents(text_dict['text'])

    for sent in sents:

        result = main_filter(sent)

        if result ==  False:

            dict['pass'].append(sent)

        else:

            dict['filted'].append((sent,result))

            # spans , mark = Split_Sent(sent)
            #
            # if mark == True:
            #
            #     for span in spans:
            #
            #         result = main_filter(span)
            #
            #         if result == False:
            #
            #             dict['pass'].append(span)
            #
            #         else:
            #
            #             dict['filted'].append((span,result))

    return dict
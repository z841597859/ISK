from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_md', disable=["ner"]) # just for lemma

def read_subjects(path):
    subjects = set()
    with open(path) as f:
        for line in f:
            subjects.add(line.strip())
    return subjects

# original quasimodo compute function, for spo-like KBs, 25.63%, 55.07%
def lemmatize_quasimodo(s):
    if "has_" in s:
        s = "be"
    doc = nlp(s)
    res = []
    for x in doc:
        res.append(x.lemma_)
    return " ".join(res)
PREDICATES_CONCEPTNET = {
    "AtLocation": "be in",
    "CapableOf": "can",
    "Causes": "cause",
    "CausesDesire": "cause desire",
    "CreatedBy": "create by",
    "DefinedAs": "define as",
    "Desires": "desire",
    "DistinctFrom": "be distinct from",
    "Entails": "entail",
    "HasA": "have",
    "HasFirstSubevent": "have",
    "HasLastSubevent": "have",
    "HasPrerequisite": "need",
    "HasProperty": "be",
    "HasSubevent": "have",
    "InstanceOf": "be",
    "LocatedNear": "be near",
    "MadeOf": "make of",
    "MannerOf": "be",
    "MotivatedByGoal": "motivate by",
    "PartOf": "be part of",
    "ReceivesAction": "receive action",
    "UsedFor": "use for",
    "has_body_part": "be",
    "has_color": "be",
    "has_diet": "be",
    "has_effect": "be",
    "has_height": "be",
    "has_manner": "be",
    "has_movement": "be",
    "has_place": "be",
    "has_property": "be",
    "has_temperature": "be",
    "has_time": "be",
    "has_trait": "be",
    "has_weather": "be",
    "has_weight": "be"
}
def read_tsv_quasimodo(quasimodo_line_generator, subjects):
    assos = dict()
    for line in tqdm(quasimodo_line_generator):
        if len(line) < 3:
            continue
        subj = line[0]
        if subj not in subjects:
            continue
        pred = line[1]
        if pred in PREDICATES_CONCEPTNET:
            pred = PREDICATES_CONCEPTNET[pred]
        pred = lemmatize_quasimodo(pred)
        obj = lemmatize_quasimodo(line[2])
        if subj in assos:
            assos[subj].add(pred)
            assos[subj].add(obj)
            assos[subj].add(pred + " " + obj)
        else:
            assos[subj] = {pred, obj}
            assos[subj].add(pred + " " + obj)
    return assos

def compute_recall_strict_quasimodo(assos, refer_line_generator):
    n_assos = 0
    n_found = 0
    for line in tqdm(refer_line_generator):
        if line[0] not in assos:
            continue
        for sentence in line[1:]:
            sentence = lemmatize_quasimodo(sentence.lower())
            n_assos += len(sentence) - len(line[0])
            maxi = 0
            for relation in assos[line[0]]:
                if line[0] in relation and len(relation) <= len(line[0]) + 3:
                    continue
                if relation in sentence:
                    if len(relation) > maxi:
                        maxi = len(relation)
            n_found += maxi
    if n_assos > 0:
        return "%.2f" % (n_found / n_assos * 100) + "%"
    else:
        return 0.0
def compute_recall_relaxed_quasimodo(assos, related_words_generator):
    n_assos = 0
    n_found = 0
    for line in tqdm(related_words_generator):
        if line[0] not in assos:
            continue
        n_assos += len(line) - 1
        for word in line[1:]:
            word = lemmatize_quasimodo(word)
            for relation in assos[line[0]]:
                if word in relation.split(" "):
                    n_found += 1
                    break
    if n_assos > 0:
        return "%.2f" % (n_found / n_assos * 100) + "%"
    else:
        return 0.0

# disassmble quasimodo compute function, for spo-like KBs, 25.63%, 55.07%
def quasimodo_hit_subjects(spokb_line, subjects):
    subj = spokb_line[0]
    if subj in subjects:
        return subj
    else:
        return False
def quasimodo_assos_main(spokb_line,subjects):
    # if len(spokb_line) < 3:
    #     return None, None
    hit_result = quasimodo_hit_subjects(spokb_line, subjects)
    if not hit_result:
        return None, None
    subj = hit_result
    pred = spokb_line[1]
    if pred in PREDICATES_CONCEPTNET:
        pred = PREDICATES_CONCEPTNET[pred]
    pred = lemmatize_quasimodo(pred)
    obj = lemmatize_quasimodo(spokb_line[2])
    return subj, [pred, obj, pred + " " + obj]

def quasimodo_assos_updating(main_result,assos):
    subj = main_result[0]
    relations = main_result[1]
    if subj in assos:
        assos[subj].update(relations)
    else:
        assos[subj] = set(relations)

def quasimodo_overlap_strcit(refer_sent, assos_one_relations_set, subj):
    maxi = 0
    for relation in assos_one_relations_set:
        if subj in relation and len(relation) <= len(subj) + 3:
            continue
        if relation in refer_sent:
            if len(relation) > maxi:
                maxi = len(relation)
    return maxi
def compute_recall_strict_quasimodo_disa(assos, refer_line_generator):
    n_assos = 0
    n_found = 0
    for line in tqdm(refer_line_generator):
        if line[0] not in assos:
            continue
        for sentence in line[1:]:
            sentence = lemmatize_quasimodo(sentence.lower())
            n_assos += len(sentence) - len(line[0])
            maxi = quasimodo_overlap_strcit(sentence, assos[line[0]], line[0])
            n_found += maxi
    if n_assos > 0:
        return "%.2f" % (n_found / n_assos * 100) + "%"
    else:
        return 0.0

def quasimodo_overlap_relaxed(refer_word, assos_one_relations_set):
    for relation in assos_one_relations_set:
        if refer_word in relation.split(" "):
            return True
    else:
        return False
def compute_recall_relaxed_quasimodo_disa(assos, related_words_generator):
    n_assos = 0
    n_found = 0
    for line in tqdm(related_words_generator):
        if line[0] not in assos:
            continue
        for word in line[1:]:
            n_assos += 1
            word = lemmatize_quasimodo(word)
            if quasimodo_overlap_relaxed(word, assos[line[0]]):
                n_found += 1
    if n_assos > 0:
        return "%.2f" % (n_found / n_assos * 100) + "%"
    else:
        return 0.0

# for sent-like KBs  oie result, whose spo is still more complex than spo-like KBs, more situations and words
def lemmatize_oie(s):
    if s == None:
        return ''
    doc = nlp(s)
    res = []
    for x in doc:
        res.append(x.lemma_) # 必小写
    return " ".join(res)
def lemma_split_oie(s):
    if s == None:
        return []
    doc = nlp(s)
    res = []
    for x in doc:
        res.append(x.lemma_) # 必小写
    return res

def qlike_hit_subjects(oie_line, subjects):
    subj = oie_line['subject']
    if subj in subjects:
        return subj
    else:
        return False
def qlike_assos_main(oie_line,subjects):
    hit_result = qlike_hit_subjects(oie_line, subjects)
    if not hit_result:
        return None, None
    subj = hit_result
    pred = oie_line['predicate']
    pred = lemmatize_oie(pred)
    obj = lemmatize_oie(oie_line['object'])
    return subj, [pred, obj, pred + " " + obj]

def hit_obj_subjects(one_oie_result, subjects):
    raw_obj = one_oie_result['object']
    raw_obj = lemma_split_oie(raw_obj)
    for obj in raw_obj:
        if obj in subjects:
            return obj
    return False
def hit_subj_subjects(one_oie_result, subjects):
    raw_subj = one_oie_result['subject']
    raw_subj = lemma_split_oie(raw_subj)
    for subj in raw_subj:
        if subj in subjects:
            return subj
    return False
def assos_main(one_oie_result, subjects):
    hit_result = hit_subj_subjects(one_oie_result, subjects)
    if hit_result != False:
        subj = hit_result
        pred = one_oie_result['predicate']
        obj = one_oie_result['object']
        pred = lemmatize_oie(pred)
        obj = lemmatize_oie(obj)
        return subj, [pred, obj, pred + ' ' + obj]
    else:
        hit_result = hit_obj_subjects(one_oie_result, subjects)
        if hit_result != False:
            obj = hit_result
            pred = one_oie_result['predicate']
            subj = one_oie_result['subject']
            pred = lemmatize_oie(pred)
            subj = lemmatize_oie(subj)
            return obj, [pred, subj, subj + ' ' + pred]
    return None, None

def hit_obj_subjects_2(one_oie_result, subjects):
    raw_obj = one_oie_result['object']
    obj = lemmatize_oie(raw_obj)
    if obj in subjects:
        return obj
    return False
def hit_subj_subjects_2(one_oie_result, subjects):
    raw_subj = one_oie_result['subject']
    subj = lemmatize_oie(raw_subj)
    if subj in subjects:
        return subj
    return False
def assos_main_2(one_oie_result, subjects):
    hit_result = hit_subj_subjects_2(one_oie_result, subjects)
    if hit_result != False:
        subj = hit_result
        pred = one_oie_result['predicate']
        obj = one_oie_result['object']
        pred = lemmatize_oie(pred)
        obj = lemmatize_oie(obj)
        return subj, [pred, obj, pred + ' ' + obj]
    else:
        hit_result = hit_obj_subjects_2(one_oie_result, subjects)
        if hit_result != False:
            obj = hit_result
            pred = one_oie_result['predicate']
            subj = one_oie_result['subject']
            pred = lemmatize_oie(pred)
            subj = lemmatize_oie(subj)
            return obj, [pred, subj, subj + ' ' + pred]
    return None, None



def one_relation_overlap(relation, refer_sent):
    if relation in refer_sent:
        return len(relation)
    overlap = 0
    for token in refer_sent.split(" "):
        if token in relation:
            overlap += len(token)
    return overlap
def overlap_strict(refer_sent, assos_one_relations_set, subj):
    maxi = 0
    for relation in assos_one_relations_set:
        if subj in relation and len(relation) <= len(subj) + 3:
            continue
        overlap = one_relation_overlap(relation, refer_sent)
        if overlap > maxi:
            maxi = overlap
    return maxi
def compute_recall_strict(assos, refer_line_generator):
    n_assos = 0
    n_found = 0
    for line in tqdm(refer_line_generator):
        if line[0] not in assos:
            continue
        for sentence in line[1:]:
            sentence = lemmatize_quasimodo(sentence.lower())
            n_assos += len(sentence) - len(line[0])
            maxi = overlap_strict(sentence, assos[line[0]], line[0])
            n_found += maxi
    if n_assos > 0:
        return "%.2f" % (n_found / n_assos * 100) + "%"
    else:
        return 0.0
"""
# This program will read several formats of Stanford EEG reports (both adults and peds). It will extract the
# impression block. Then it will score each sentence in the impression block. A large negative score indicates that
# the sentence is more likely to be describing the presence of a seizure . A more positive score would indicate a
# less likely presence of or the absence of seizure.
#
# The dict folder contains several yaml files that use contain words which are assigned scores that make the seizure
# more or less likely.

Packages are easily installed via conda. However special instructions pertain to spacy's installation:
 First install microsoft visual studio version 25123 and the python package vs2015_runtime: 14.0.25123-0 
 Then create a new environment with conda create --name py36 python=3. Then activate that environment. 
 Then pip install -U spacy followed by python -m spacy download en
If you are using Pycharm, you can't install spacy. You must change the python interpreter in the settings menu to the
 py36 environment that you've just created. 

2 csv files are required: SHC_ALL_NOTES_FILTERED.csv which has all the eeg procedure notes 
NKdatabaseDump.csv which contains all the start and end times of each eeg file with their eeg exam number. 
I've uploaded them to the patient_reports folder on stanford's box account
"""

from __future__ import print_function, division, absolute_import

import sys
import csvkit as csv
import csv
import re
import string
from pprint import pprint
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import Tree
from spacy.en import English
from spacy.symbols import nsubj, VERB
import yaml
import sys
import os
import re
import datetime
import pandas as pd
import numpy as np
from IPython.display import display
import codecs

# *******************SETTINGS********************
# input file name - a csv formatted file. one of the columns should be labelled as "note" and contain the entire
# eeg report.
input_file_name = "SHC_ALL_NOTES_FILTERED.csv"
# output file name - this file is the same as the input file with several additional columns
# There will be a column "impression" which is the extracted impression block from the notes
# There will be an notetype, duration, impressionType (sz, ed, dc), seizureScore, etc.
out_file = open("SHC_ALL_NOTES_IMPRESS_.csv", 'w')
# A csv file containing 3 columns: INF_ScheduleExam@ExamNo_STR	INF_ScheduleExam@StartTime_DT	INF_ScheduleExam@EndTime_DT
# This file is used for looking up the duration of the exam based on NK's database. The duration on the reports are
# not as reliable or uniform. The times on this CSV file are directly obtained from downloading the NK database
nkDurationFilename = "NKdatabaseDump.csv"
# ***********************************************



# all real expression patterns below for extracting the impression block
eegImprPattern = r'(?:\
Impression and Summary:|IMPRESSION and SUMMARY:|IMPRESSION and CONCLUSIONS:\
|IMPRESSION|IMPRESSIONS|IMPRESSION:|Impression:|IMPRESSIONS:\
|INTERPRETATIONS:|INTERPRETATION:|Interpretation:\
|CONCLUSION:|CONCLUSIONS:|Conclusion:\
|Summary:|SUMMARY:)\s*(?P<eegno>[\w\d \.\-\(\),]+)'
re_eegno = re.compile(eegImprPattern, re.DOTALL | re.MULTILINE)
# re_eegno is pretty specific for identifying impression blocks
# re_eegnoLoose is less specific and is tried after the re_eegno fails
eegImprPattern = r'(?:\
Impression and Summary:|IMPRESSION and SUMMARY:|IMPRESSION and CONCLUSIONS:\
|IMPRESSION|IMPRESSIONS|IMPRESSION:|Impression:|IMPRESSIONS:|Impression|impression:\
|INTERPRETATIONS:|INTERPRETATION:|Interpretation\
|CONCLUSION:|CONCLUSIONS:|Conclusion:\
|Summary:|SUMMARY:|SUMMARY|Summary)\s*(?P<eegno>[\w\d \.\-\(\),]+)'
re_eegnoLoose = re.compile(eegImprPattern, re.DOTALL | re.MULTILINE)

# RE for detecting EEG numbers. this will be important for cross referencing the NK database for durations that were not
# mentioned on the procedure note
eegTypePattern = r'[a-z]{1}[0-9]+\-[0-9]+'
re_eegnoType = re.compile(eegTypePattern, re.DOTALL | re.IGNORECASE)

# a strict to loose list of regular expressions to detect of duration
eegDurationPattern = r'Duration:\s*([\d]+:[\d]+:[\d]+)'
re_eegnoDuration = re.compile(eegDurationPattern, re.DOTALL | re.IGNORECASE)
eegDurationPattern2 = r'Duration of Study:\s*(([\d]+)\s*hour|([\d]+)\s*day)'
re_eegnoDuration2 = re.compile(eegDurationPattern2, re.DOTALL | re.IGNORECASE)
eegDurationPattern3 = r'test dates:\s*([\d\/]+)\s*-\s*([\d\/]+)'
re_eegnoDuration3 = re.compile(eegDurationPattern3, re.DOTALL | re.IGNORECASE)
eegDurationPattern4 = r'\s*([\d\.]+)\s*(hour(s?)|day(s?))'
re_eegnoDuration4 = re.compile(eegDurationPattern4, re.DOTALL | re.IGNORECASE)
eegDurationPattern5 = r'test dates:\s*([\d\/]+).*?-\s*([\d\/]+)'
re_eegnoDuration5 = re.compile(eegDurationPattern5, re.DOTALL | re.IGNORECASE)

# detects commonly used phrases in the impression
eegAbnormalityTypePattern = ['no epileptiform', 'absence of epileptiform', 'not epileptiform']
# r'(?<!no)\s+(epileptiform)'
eegAbnormalityTypePattern2 = r'(?<!absence of)\s+(epileptiform)'






def tsvTOcsv(input_file_name, out_file):
    """
    converts tsv formatted files to csv,
    used to make the csv file that is readable by this EEG report feature analyzer
    :param input_file_name: a tsv file
    :param out_file: a csv file
    :return: 
    """
    i = 0
    with open(input_file_name, 'rb') as tsvin, open(out_file, 'wb') as csvout:
        tsvin = csv.reader(tsvin, delimiter='\t')
        csvout = csv.writer(csvout)

        for row in tsvin:
            if len(row) > 0:
                csvout.writerow(row)


def findTrueImpression(eeg_no, line, i):
    # an early attempt to categorize impressions which did not work
    wordlist = eeg_no.lower().split()
    words = [''.join(c for c in s if c not in string.punctuation) for s in wordlist]
    if "normal" in words and "abnormal" in words:
        # equivocal
        result = findSpecificPhrases(eeg_no, line, i)
        if (result == False):
            line['impression'] = "unknown"
            print(repr(i) + "EQUIVOCAL" + eeg_no)
            print()

    elif "abnormal" in words:
        line['impression'] = "abnormal"
    elif "status" in words:
        line['impression'] = "abnormal"
    elif "seizure" in words and "nonepileptic" in words:
        result = findSpecificPhrases(eeg_no, line, i)
        if (result == False):
            line['impression'] = "unknown"
            print(repr(i) + "UNKNOWN-" + eeg_no)
            print()
    elif "nonepileptic" in words:
        line['impression'] = "normal"
    elif "seizure" in words:
        line['impression'] = "abnormal"
    elif "slowing" in words:
        line['impression'] = "abnormal"
    elif "normal" in words:
        line['impression'] = "normal"
    else:
        line['impression'] = eeg_no
        # print("(" + repr(i) + ")" + line['impression'] + "-" + eeg_no.lower())


def findSpecificPhrases(eeg_no, line, i):
    # finds specific commonly used phrases and categorizes the impressionType
    # also did not work well
    if "This EEG is normal" in eeg_no:
        line['impression'] = "normal"
    elif "is normal for age" in eeg_no:
        line['impression'] = "normal"
    elif "EEG is within the normal" in eeg_no:
        line['impression'] = "normal"
    elif "EEG is within the broad normal" in eeg_no:
        line['impression'] = "normal"
    elif "EEG recording was normal" in eeg_no:
        line['impression'] = "normal"
    elif "This record is normal" in eeg_no:
        line['impression'] = "normal"
    elif "EEG recording is normal" in eeg_no:
        line['impression'] = "normal"
    elif "This is a normal" in eeg_no:
        line['impression'] = "normal"


    elif "This EEG is abnormal" in eeg_no:
        line['impression'] = "abnormal"
    elif "is abnormal because of" in eeg_no:
        line['impression'] = "abnormal"
    elif "recording is abnormal" in eeg_no:
        line['impression'] = "abnormal"
    elif "markedly abnormal" in eeg_no.lower():
        line['impression'] = "abnormal"
    elif "is abnormal due to" in eeg_no:
        line['impression'] = "abnormal"
    elif "This record is abnormal" in eeg_no:
        line['impression'] = "abnormal"
    elif "This is an abnormal" in eeg_no:
        line['impression'] = "abnormal"
    else:
        return False

def setImpressionBody(m, line, i):
    # finds the body in the procedural note. useful for vague impressions
    impstr = line['note'][m.start():]
    n = re.search(r'(?:comments|comment|clinical correlation|DETAILED FINDINGS)', impstr, re.IGNORECASE)
    if n:
        line['impressionBody'] = impstr[:n.start()]
    else:
        line['impressionBody'] = impstr
    # get rid of double spaces
    line['impressionBody'] = " ".join(line['impressionBody'].split())


def setImpressionType(line, i):
    # early attempt to use commonly used phrases for categorizing impressionType
    line['impressionType'] = ""
    impstr = line['impressionBody']
    if impstr is None:
        return
    # https://regex101.com/

    n = re.search(r'(?:(?<!absence of)(?<!no)(?<!not))\s+(epileptiform)', impstr, re.IGNORECASE)
    if n:
        if ("other epileptiform" not in impstr.lower()):
            line['impressionType'] += ' ed'  # epileptiform discharge

    n = re.search(r'(?:(?<!absence of)(?<!no)(?<!not))\s+(discharge|spike)', impstr, re.IGNORECASE)
    if n:
        line['impressionType'] += ' dc'  # discharges which are more questionable

    n = re.search(r'seizure', impstr, re.IGNORECASE)
    if n:
        if ("no evidence for electrographic seizures" not in impstr.lower()) and \
                ("for a seizure disorder" not in impstr.lower()) and \
                ("no evidence for electrographic seizure" not in impstr.lower()) and \
                ("does not exclude a clinical diagnosis of seizure" not in impstr.lower()) and \
                ("no electrographic seizure" not in impstr.lower()):
            line['impressionType'] += ' sz'


def durationByDates(t1, t2):
    """
    # calculates the duration between two given dates t1 and t2 in the format of month/day/year like 03/30/16
    # returns day:hour:min such as 5:35:53    
    :param t1:  string format of a date 
    :param t2:  string format of a date 
    :return: string format of the duration
    """
    dtformat = '%m/%d/%Y'
    try:
        t1 = datetime.datetime.strptime(t1, dtformat)
        t2 = datetime.datetime.strptime(t2, dtformat)
        t1 = t1.date()
        t2 = t2.date()
        td = t2 - t1
        return (
            repr(td.days * 24) + ":" + repr(td.seconds // 3600).zfill(2) + ":" + repr((td.seconds // 60) % 60).zfill(2))
    except:
        return ""

def durationByHoursHopefully(eegReport):
    """
    # last ditch effort to match any mention of days or hours in the report
    # note that it returns a string represented by hours:00:00
    :param eegReport: entire eeg report
    :return: 
    """
    hours = 0
    days = 0
    i = 0
    for match in re.finditer(eegDurationPattern4, eegReport):
        matchtxt = match.group(2)
        i += 1
        if "hour" in matchtxt:
            hours = int(float(match.group(1)))
        elif "day" in matchtxt:
            days = int(float(match.group(1)))
    totalhours = hours + 24 * days
    return repr(totalhours) + ':00:00'


class Splitter(object):
    """
        # a very annoying sentence in the compound sentence in the impression. we have to split those sentences so the first
        # negative word does not apply to the second sentence too. ie: no epileptiform discharges were seen, and there was
        # one seizure. 
        input: a paragraph 
        output: a paragraph with only simple sentences 
    """

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        text = Splitter.splitCompoundSentences(text)
        text = text.replace("There", ". There").replace("The", ". The")
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        # instead, split compound sentences to 2 sentences
        # pprint(tokenized_sentences )
        return tokenized_sentences

    def splitCompoundSentences(paragraph):
        parsedData = spacyParser(paragraph)

        sents = ""
        # the "sents" property returns spans
        # spans have indices into the original string
        # where each index value represents a token
        for span in parsedData.sents:
            # go from the start to the end of each span, returning each token in the sentence
            # combine each token using join()
            sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
            subj_flag = False
            verb_flag = False
            compound_flag = False
            first = True
            candidate_sent = ""
            i = 0
            lastPeriod = 0
            # if the phrase we have encountered so far has a verb noun and a punctuation then it's a sentence
            for token in span[:-1]:
                # print(str(token.orth_), str(token.pos_))
                if token.pos_ == "VERB" and (subj_flag == True):
                    verb_flag = True
                    # print("verb:"+str(token.orth_))
                if (token.pos_ == "NOUN"):
                    subj_flag = True
                    # print("subj:"+str(token.orth_))
                if token.pos_ == "PUNCT":
                    if (subj_flag == True) and (verb_flag == True):
                        # compound sentence detected; split the sentence
                        subj_flag = False
                        verb_flag = False
                        candidate_sent = str(span[lastPeriod:i - 1]) + "."
                        # candidate_sent = candidate_sent[0:len(candidate_sent)-1]
                        lastPeriod = i + 1
                        # candidate_sent += (".")
                        sents += candidate_sent + "\n"
                        # candidate_sent = ""

                i += 1
                # candidate_sent += str(token.orth_) + " "

                # print(candidate_sent)

                # print("HI" + str(candidate_sent))
            candidate_sent = str(span[lastPeriod:])
            # candidate_sent = candidate_sent[0:len(candidate_sent)-1]
            # candidate_sent += str(span[-1])
            sents += candidate_sent + "\n"

        # flatten the list into a paragraph for nltk
        # paragraph = [item for sublist in sents for item in sublist]
        # paragraph = ' '.join(map(str, paragraph))
        # print(sents)
        return sents


# POSTagger will label each word as a verb, noun, etc.
class POSTagger(object):
    def __init__(self):
        pass

    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        # adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos


class DictionaryTagger(object):
    """
    input: posttagged sentence from above
    output: posttagged sentences with numbers assigned
        the numbers represent inversion, badness, etc --> when calculated later on, it becomes a score represeting 
        the probability of a seizure / ed
    """

    def __init__(self, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        for file in files:
            file.close()
            # map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))
                    # self.closure(files)

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N)  # avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    # self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token:  # if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

        # def __del__(self):
        # print("closing dict")
        # self.closure


def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0


def sentence_score_deprecated(sentence_tokens, previous_token, acum_score):
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)


def sentence_score(sentences):
    scored_sentences = []
    if not sentences:
        return None
    else:
        for sentence in sentences:
            tagged_expression = []
            token_score = 0
            for (word, wordstem, postag) in sentence:
                if 'inc' in postag:
                    token_score *= 2.0
                elif 'dec' in postag:
                    token_score /= 2.0
                elif 'inv' in postag:
                    token_score += -1
                else:
                    token_score = 0
                tagged = (word, wordstem, postag, token_score)
                tagged_expression.append(tagged)
            scored_sentences.append(tagged_expression)
    return scored_sentences


def sentence_simplify(sentences):
    """
    given a paragraph, remove words that are not important for the meaning of the sentence
    returns a paragraph of phrases.
    """

    scored_sentences = []
    if not sentences:
        return None
    else:
        for sentence in sentences:
            tagged_expression = []
            for (word, wordstem, postag) in sentence:
                # print(postag)
                if ('DT' in postag) or \
                        ('IN' in postag) or \
                        ('NN' in postag) or \
                        ('RB' in postag) or \
                        ('JJ' in postag) or \
                        ('NNS' in postag):
                    tagged = (word, wordstem, postag)
                    tagged_expression.append(tagged)
                    # print('ok')
            if tagged_expression:
                scored_sentences.append(tagged_expression)
    return scored_sentences


def sentiment_score(review):
    """    
        The original sentiment score calculator for movie reviews, kept as a reference
    """
    finalized_score = 0
    if (review is None):
        return 0
    for sentence in review:
        if sentence:
            # print()
            # pprint(sentence)
            token_score = 0
            for (word, wordstem, tags, score) in sentence:
                token_score += sum([value_of(tag) for tag in tags])
            # print(token_score)
            for (word, wordstem, tags, score) in sentence:
                if score != 0:
                    token_score *= score
                    #        print(word + str(score) +"-->"+str(token_score))
            # in examples of 'there are seizures. there are events with no ictal pattern', we have to ignore the next sentence
            # print("score is now " + str(token_score))
            if (token_score < 1):
                finalized_score += token_score
                #    print("final: " + str(finalized_score))
    return finalized_score


def sentimentAnalysisForSeizure(text):
    """
        sentiment score was originally written to determine the negativity or positivity of movie review articles. i'm 
        using it for determining how likely the impression is indicating a seizure. First paragraphs are split into
        simple sentences. Then they are tagged by grammatical structures. Then useless words are removed. Then each
        word is scored. Then the scores are added up to a final score indicating the probability of the paragraph 
        indicating that there is a seizure. 
    """

    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger(['dicts/seizure/positive.yml', 'dicts/seizure/negative.yml',
                                   'dicts/seizure/inc.yml', 'dicts/seizure/dec.yml', 'dicts/seizure/inv.yml'])

    if isinstance(text, str):
        splitted_sentences = splitter.split(text)
    else:
        return 0
    # pprint(text)
    # pprint(splitted_sentences)

    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    # pprint(pos_tagged_sentences)

    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # pprint(dict_tagged_sentences)

    dict_tagged_sentences = sentence_simplify(dict_tagged_sentences)
    # print("simplified:", "")
    # print(dict_tagged_sentences)
    # () + 1
    score_tagged_sentences = sentence_score(dict_tagged_sentences)
    # print("scored:", "")
    # pprint(score_tagged_sentences)
    # print("analyzing sentiment...")
    score = sentiment_score(score_tagged_sentences)
    # print(score)
    return score


def sentimentAnalysisForEpileptiform(text):
    """
        Just like the sentiment analyzer above except determines the likelihood of detecting epileptiform discharges
        which are equivalent to discharges
    :param text: the impression block
    :return: a score indicating the likelihood of seizures. anything less than 0 is considered positive for seizure
    """

    splitter = Splitter()
    postagger = POSTagger()
    dicttagger = DictionaryTagger(['dicts/epileptiform/positive.yml', 'dicts/epileptiform/negative.yml',
                                   'dicts/epileptiform/inc.yml', 'dicts/epileptiform/dec.yml',
                                   'dicts/epileptiform/inv.yml'])

    if isinstance(text, str):
        text = text.replace("-", " ")
        splitted_sentences = splitter.split(text)
    else:
        return 0
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    # pprint(pos_tagged_sentences)
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    del dicttagger
    dict_tagged_sentences = sentence_simplify(dict_tagged_sentences)
    score_tagged_sentences = sentence_score(dict_tagged_sentences)
    # print("scored:", "")
    # pprint(score_tagged_sentences)
    score = sentiment_score(score_tagged_sentences)
    return score


#import en_core_web_sm

# conda
# nlp = en_core_web_sm.load()

# some tests
# sentimentAnalysisForSeizure("IMPRESSION: This EEG capturing wakefulness and sleep is ABNORMAL due to: 1) mild diffuse slowing 2) multifocal and generalized spikes and spike-and-wave complexes (including slow spike wave) and occasional diffuse paroxysmal fast activity 3) ~4 brief <20 second seizures/ictal patterns (without logged behaviors) Overall, given that the patient has had similarly abnormal EEGs in the remote past, occurrence of four <20 second does not explain decreased oral intake and increased sleepiness.")
# output = sentimentAnalysisForEpileptiform(
#    "IMPRESSION: This is an abnormal ambulatory EEG due to bursts of generalized polyspike and polyspike wave. There are no prolonged runs or seizures during this recording.")
# sentimentAnalysisForEpileptiform("INTERPRETATION: This is an abnormal study due to the presence of occasional to frequent bursts of slowing, mostly diffuse and rarely more lateralized to the left or right hemisphere. There were no clear epileptiform discharges or seizures. Comments The bursts of slowing are overall non-specific with regards to etiology. This finding may indicate diffuse cerebral dysfunction. There is no clear indication of epileptiform discharges, although a rhythmic epileptiform abnormality related to the bursts of slowing cannot entirely be excluded. Absence of epileptiform discharges does not preclude a clinical diagnosis of epilepsy or seizures. Clinical correlation is advised.")
# sentimentAnalysisForSeizure("INTERPRETATION: Impression This is a normal awake and asleep EEG. There are no seizures or epileptiform discharges during this recording. Comments Absence of epileptiform discharges does not exclude a clinical diagnosis of seizures or epilepsy. There is no documentation of patient's typical events concerning for seizures on the log sheet.")
# sentimentAnalysis("there are no such things as ictal activity abnormal")
# sentimentAnalysis("there questionable activity abnormal")
# output = sentimentAnalysisForSeizure("IMPRESSION: This EEG shows no epileptiform discharges, and there were no seizures either. Incidentically, there were seizures and epileptiform discharges")
# print(output)

def loadMasterCSVFile():
    """
    loads the durations of all eegs from an NK database derived csv file of eeg numbers with their associated start 
    and end times. 
    :return: 
    """
    with codecs.open(nkDurationFilename, 'r', encoding='utf-8', errors='ignore') as cf:
        i = 0
        masterCSV = csv.reader(cf)
        masterCSV_list = [row for row in masterCSV]
        print(len(masterCSV_list))
        return masterCSV_list


def durationByDatesForCSV(t1, t2):
    """
    
    :param t1: start time of the eeg from the csv file
    :param t2: end time of the eeg from the csv file
    :return: human readable duration string 
    """
    try:
        dtformat = '%m/%d/%Y %I:%M:%S %p'
        # try:
        # print(t1)
        t1 = datetime.datetime.strptime(t1, dtformat)
        t2 = datetime.datetime.strptime(t2, dtformat)
    # print(t2)
    # t1 = t1.date()
    # t2 = t2.date()
    except ValueError:
        try:
            dtformat = '%m/%d/%Y'
            t1 = datetime.datetime.strptime(t1, dtformat)
            t2 = datetime.datetime.strptime(t2, dtformat)
        except ValueError:
            try:
                dtformat = '%m/%d/%Y (%I:%M)'
                t1 = datetime.datetime.strptime(t1, dtformat)
                t2 = datetime.datetime.strptime(t2, dtformat)
            except:
                return None
        except:
            return None
    except:
        return None
    td = t2 - t1
    td = abs(td)
    return str(td)

def masterCSV_daterange(examNO):
    """
    return a dateDelta for any given exam number by looking up all the rows of nkDurationFilename file with an exam number,
    then looking for the minimal and maximal dates within those rows
    :param examNO: exam number
    :return: time object
    """
    df = pd.DataFrame(masterCSV_list[1:],
                      columns=masterCSV_list[0])
    po = df['INF_ScheduleExam@ExamNo_STR'].str.contains(examNO)
    resultingDF = df.iloc[np.flatnonzero(po)]
    dateList = list()
    dateList.extend(resultingDF['INF_ScheduleExam@StartTime_DT'].tolist())
    dateList.extend(resultingDF['INF_ScheduleExam@EndTime_DT'].tolist())
    if (len(dateList) is 0):
        return None
    max_value = max(dateList)
    min_value = min(dateList)
    duration = durationByDatesForCSV(min_value, max_value)
    return duration


# test
# masterCSV_list = loadMasterCSVFile()
# masterCSV_daterange("A17-293")

def analyzeAllEEG():
    """
    reads the entire table of eeg reports (input_file_name) which is a csv file and creates a new csv file with new columns
    which will include the likelihood score of it containing seizures/ed. 
    :return: nothing. this function writes to the outfile
    """
    i = 1
    with open(input_file_name) as cf:
        # needed to replace null lines
        reader = csv.DictReader(x.replace('\0', '') for x in cf)
        # optional starting offset in case program crashes while analyzing
        # for line in reader:
        #    i += 1
        #    if (i>9425):
        #        i=1
        #        break

        outfieldnames = reader.fieldnames
        outfieldnames.append('notebody')        # body of the procedure note, excludies impression
        outfieldnames.append('examno')          # exam number ie A033
        outfieldnames.append('impressionBody')  # impression block
        outfieldnames.append('impression')      # just says whether or not the eeg report was abnormal or normal
                                                # the algorithm only looks at specific commonly used phrases
        outfieldnames.append('notetype')        # spot v ambulatory v long term monitoring...
        outfieldnames.append('duration')        # duration in the format of Days, Hours:Minutes:Seconds
        outfieldnames.append('impressionType')  # ed, dc, sz - marks each report with these tags based of the sentiment score
        # next two column of features are based of a better algorithm that will try to understand the syntax of each
        # sentence instead of just looking for specific phrases (dubbed "sentiment" analysis)
        outfieldnames.append('epileptiformScore')
        outfieldnames.append('seizureScore')
        writer = csv.DictWriter(out_file, fieldnames=outfieldnames, restval='*')
        writer.writeheader()

        for line in reader:
            i += 1
            eeg_no = ""
            m = re_eegno.search(line['note'])
            # write the notebody and impression
            # impressions are determined by matching for specific phrases
            # this method was not very reliable
            if m:
                line['notebody'] = line['note'][:m.start()]
                eeg_no = m.group('eegno')
                findTrueImpression(eeg_no, line, i)
                setImpressionBody(m, line, i)
            else:
                # try a looser find that may introduce more false information (more sensitive less specific pattern)
                m = re_eegnoLoose.search(line['note'])
                if m:
                    line['notebody'] = line['note'][:m.start()]
                    eeg_no = m.group('eegno')
                    findTrueImpression(eeg_no, line, i)
                elif "preliminary" in line['note'].lower():
                    line['impression'] = "prelim"
                elif "prelim" in line['note'].lower():
                    line['impression'] = "prelim"
                #else:
                    #print("###IMPRESSION BLOCK MISSING-" + repr(i) + line['note'])
                    #print()

            # figure out the duration by looking up the duration of the study based on the exam number from NK database
            m = re_eegnoType.search(line['note'])
            line['duration'] = None
            if m:
                line['duration'] = masterCSV_daterange(m.group(0))
            if line['duration'] is None:
                # write the duration by some other messier means through regular expression matching
                m = re_eegnoDuration.search(line['note'])
                if m:
                    match = m.group(1)
                    line['duration'] = match
                else:
                    m = re_eegnoDuration2.search(line['note'])
                    if m:
                        match = m.group(1)
                        if "hour" in match:
                            line['duration'] = m.group(2) + ':00:00'
                        elif "day" in match:
                            line['duration'] = repr(int(m.group(3)) * 24) + ':00:00'
                    else:
                        m = re_eegnoDuration3.search(line['note'])
                        if m:
                            line['duration'] = durationByDates(m.group(1), m.group(2))
                        else:
                            m = re_eegnoDuration4.search(line['note'])
                            if m:
                                line['duration'] = durationByHoursHopefully(line['note'])
                            else:
                                m = re_eegnoDuration5.search(line['note'])
                                if m:
                                    print(m.group(1))
                                    line['duration'] = durationByDates(m.group(1), m.group(2))

            # write the notetype
            m = re_eegnoType.search(line['note'])
            if m:
                match = m.group(0)
                match = match.lower()
                line['examno'] = match
                if ("s" in match):
                    line['notetype'] = "spot"
                elif ("v" in match):
                    line['notetype'] = "ceeg"
                elif ("a" in match):
                    line['notetype'] = "ambu"
                elif ("f" in match):
                    line['notetype'] = "spot inpt"
                elif ("e" in match):
                    line['notetype'] = "ceeg"
                else:
                    line['notetype'] = "unk"

            # determine procedure type by duration
            # make sure ambulatories are >24
            dtformat = '%H:%M:%S'
            try:
                t = datetime.datetime.strptime(line['duration'], dtformat)
                totMinutes = t.minute + t.hour * 60
                # determine the unknown report based off the duration only
                if line['duration'] is not None:
                    if line['notetype'] is None:
                        if totMinutes < 100 and totMinutes > 18:
                            line['notetype'] = "spot?"
                            pass
                if "day" not in line['duration']:
                    if "ambu" is line['notetype']:
                        if totMinutes < 1320 and totMinutes > 1:
                            line['notetype'] = "longspot"
            except:
                pass

            # write sentiment score
            line['seizureScore'] = sentimentAnalysisForSeizure(line['impressionBody'])
            line['epileptiformScore'] = sentimentAnalysisForEpileptiform(line['impressionBody'])

            # write the abnormality type
            line['impressionType'] = ""
            if int(line['seizureScore']) < 0:
                line['impressionType'] += ' sz'
            if int(line['epileptiformScore']) < 0:
                line['impressionType'] += ' ed'

            writer.writerow(line)
            # Since i'm just concerned about ambulatory reports, i'll just only include ambulatory type eegs
            # In the future, this line can be commented out
            # if (line['notetype'] is "ambu"):
                # writer.writerow(line)
                # print(type(line['Note']))
                # print("line %s" + line['note_Id'][:20] + repr(line['sentimentScore']), i )
                # print("line %s" + line['notebody'][:200], i )
                # print()
                # if i > 1555:
                #    break
                # print(eeg_no)
                # if (i>600):
                #    break
    #close(input_file_name)
    #writer.close()






# initialize the spacyParser which is only being used to determine compound sentences
spacyParser = English()

# initialize the lookup table of eeg durations
masterCSV_list = loadMasterCSVFile()

analyzeAllEEG()


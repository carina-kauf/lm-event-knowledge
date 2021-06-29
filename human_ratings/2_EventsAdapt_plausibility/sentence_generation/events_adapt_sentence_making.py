import pandas as pd
import numpy as np
import re
import collections

################################
# Producing original sentences #
################################

orig_df = pd.read_excel("adapt-realmaterials.xls")
orig_df = orig_df.loc[orig_df["Sub-expt"] != "GlobSem"]  # not using GlobSem

# making a df of sentences (so we don't lose data like Lex, AAR)
sent_sofar = set()
sent_data = []

for idx in orig_df.index:
    sent = orig_df.at[idx, 'S1']
    if sent not in sent_sofar:
        sent_data.append({
            'original_sentence': sent,
            'sub_expt': orig_df.at[idx, 'Sub-expt'],
            'evtype': orig_df.at[idx, 'EvType'],
            'voice': ('Passive' if re.search(r'\bwere\b|\bwas\b', sent)
                      else 'Active'),
        })
        sent_sofar.add(sent)

sent_df = pd.DataFrame(sent_data)


# # Making a list of sentences for original data, simpler version for testing
# s1 = list(pd.unique(orig_df["S1"]))
# s2 = list(pd.unique(orig_df["S2"]))
# orig_sent = s1 + [s for s in s2 if s not in s1]
# # check for article issues
# weird_sent = [sent for sent in orig_sent
#               if len(re.findall(r'\bthe\b', sent, re.IGNORECASE)) != 2
#               or re.findall(r'\ba\b', sent, re.IGNORECASE)]

# making changes based on issues with articles
changes_dict = {r"\bhis\b": "the", r"\ba\b": "the", r"\bA\b": "The"}
# convert articles
change_words = ['jam', 'marmalade', 'jail', 'prison']
changes_dict.update({fr"\b{word}\b": f"the {word}" for word in change_words})
changes_dict.update({fr"\b{word[0].upper()}{word[1:]}\b": f"The {word}"
                     for word in change_words})
# convert 'jam' -> 'the jam' and 'Jam' -> 'The jam'
changes_dict[r'\bsautéed\b'] = 'sauteed'  # avoid char issues
changes_dict[r'\bmode\b'] = 'model'  # typo


# additions to round things out
sent_changes = {
    'The runner bumped into the co-worker.':   'The runner encountered the co-worker.',
    'The jogger ran into the colleague.':      'The jogger met the colleague.',
    'The perpetrator ratted out the poacher.': 'The perpetrator exposed the poacher.',
    'The proprietor kicked out the renter.':   'The proprietor dislodged the renter.',
    'The co-worker was bumped into by the runner.':   'The co-worker was encountered by the runner.',
    'The colleague was run into by the jogger.':      'The colleague was met by the jogger.',
    'The poacher was ratted out by the perpetrator.': 'The poacher was exposed by the perpetrator.',
    'The renter was kicked out by the proprietor.':   'The renter was dislodged by the proprietor.',
}

pattern = '|'.join(changes_dict)

initial_updates = sent_df['original_sentence'].apply(
    lambda s: re.sub(pattern, lambda m: changes_dict[fr"\b{m.group(0)}\b"], s))
# sent_df.update(pd.DataFrame(initial_updates))
initial_updates = initial_updates.replace(sent_changes)
initial_updates = initial_updates.rename("updated_sentence")

sent_df = pd.concat([sent_df, initial_updates], axis=1)
# with open("adapt_orig_sentences.txt", "w") as f:
#     f.writelines(f"{s}\n" for s in initial_updates)

# for i, sent in enumerate(orig_sent):
#     orig_sent[i] = re.sub(pattern,
#                           lambda m: changes_dict[fr"\b{m.group(0)}\b"],
#                           sent)

##########################################
# Parsing original sentences with checks #
##########################################

# with open("adapt_orig_sentences.txt", "r") as f:
#     orig_sent = [s.strip('\n') for s in f.readlines()]

verbs_list = list(pd.read_csv("EventsAdapt_unique_sentence_data.csv")["Verb"])
verbs_list += ['overheard', 'dislodged', 'met']
# loads parsed verbs from old sheet, might have errors


def parse_active(sentence):
    """
    Takes in a standard active sentence string, returns list of noun and
    verb phrases in order [np1, np2, vp]
    Standard form: "The NP1 VP the NP2"
    """
    sub, obj = sentence.removeprefix("The ").split(' the ')
    np2 = obj.strip('.').strip()
    np1 = None
    for verb in verbs_list:
        match = re.search(fr"\b{verb}\b", sub)
        if match:
            verb_start = match.span()[0]  # if verb not in list, set to 0
            vp = sub[verb_start:].strip()
            np1 = sub[:verb_start].strip()
            break
    if not np1:  # if no verb match, verb_start = 0 and so np1 is empty
        raise ValueError(f"No verb match for {sentence}")
    return [np1, np2, vp]


def parse_passive(sentence):
    """
    Takes in a standard passive sentence string, returns list of noun and
    verb phrases in order [np1, np2, vp]
    Standard form: "The NP1 was/were VP by the NP2"
    """
    sub, obj = sentence.split('by the')
    np2 = obj.strip('.').strip()
    if re.search(r'\bwere\b', sub):  # contains the word "were"
        splitter = ' were '
    else:
        splitter = ' was '
    np1 = sub.split(splitter)[0].removeprefix('The').strip()
    vp = sub.split(splitter)[1].strip()
    return [np1, np2, vp]


issues_list = []
parsed_data = {}
for idx in sent_df.index:
    sent = sent_df.at[idx, 'updated_sentence']
    try:
        phrases = (parse_passive(sent) if sent_df.at[idx, 'voice'] == 'Passive'
                   else parse_active(sent))
        parsed_data[idx] = phrases + [len(p.split()) == 1 for p in phrases]
    except ValueError:
        issues_list.append(sent)

phrase_names = ["NP1", "NP2", "VP"]
column_names = phrase_names + [f"{p}_check" for p in phrase_names]
parsed_df = pd.DataFrame.from_dict(parsed_data, orient='index',
                                   columns=column_names)

sent_df = pd.concat([sent_df, parsed_df], axis=1)
# sent_df.to_csv('test_parsed_data.csv')

sent_df = sent_df.loc[sent_df['VP_check'] == True]


def match_series(term, series):
    """
    Returns the list of indices of series which match term exactly
    """
    s = series.str.match(term)
    return list(s.index[s])


sent_df['NP1_crossmatch'] = sent_df['NP1'].apply(
    lambda x: match_series(x, sent_df['NP2']))
sent_df['NP2_crossmatch'] = sent_df['NP2'].apply(
    lambda x: match_series(x, sent_df['NP1']))
# NP1_crossmatch crosschecks which terms in NP2 match each term in NP1
sent_df['VP_match'] = sent_df['VP'].apply(
    lambda x: match_series(x, sent_df['VP']))


def common_elem(lists, n=None):
    """
    Returns some element which appears in n of the lists
    """
    if n is None:
        n = len(lists)
    for elem in set().union(*lists):  # considers elements in all lists
        if sum([(elem in l) for l in lists]) >= n:  # count no of sets elem in
            return elem
    return None  # if no common elem.


sent_df['match'] = sent_df.apply(
    lambda x: common_elem([x.NP1_crossmatch, x.NP2_crossmatch, x.VP_match], 2),
    axis=1)
sent_df['matched_sentence'] = sent_df.apply(
    lambda x: sent_df.loc[x.match, 'updated_sentence'], axis=1)

# sent_df.to_csv('test_matched_data.csv')

matched_df = sent_df.loc[sent_df['voice'] == 'Active']

active_irreg = ['gave', 'ate', 'took', 'drank', 'blew', 'ran', 'drove', 'saw',
                'sang', 'burned', 'hid', 'chose', 'beat', 'stole', 'forgot']
passive_irreg = ['given', 'eaten', 'taken', 'drunk', 'blown', 'run',
                 'driven', 'seen', 'sung', 'burnt', 'hidden', 'chosen',
                 'beaten', 'stolen', 'forgotten']
irreg_dict = {active_irreg[k]: passive_irreg[k] for k in
              range(len(active_irreg))}  # dict of irregular verb conversions


def generate_sentences(input_series):
    """
    Takes in a pandas series (with NP1, NP2 and VP columns), outputs a series
    of 4 versions of the sentence. (active/passive and also flipped NPs)
    Assumes input is Active
    """
    output_dict = {}
    NP1, NP2, VP1 = input_series.NP1, input_series.NP2, input_series.VP
    VP2 = irreg_dict.get(VP1, VP1)  # if VP1 not in irreg_dict, VP2 = VP1
    prep = ('were' if re.search(r'\bwere\b', input_series.matched_sentence)
            else 'was')
    output_dict["active_original"] = f"The {NP1} {VP1} the {NP2}."
    output_dict["active_reversed"] = f"The {NP2} {VP1} the {NP1}."
    output_dict["passive_original"] = f"The {NP2} {prep} {VP2} by the {NP1}."
    output_dict["passive_reversed"] = f"The {NP1} {prep} {VP2} by the {NP2}."
    return pd.Series(output_dict)


new_sentences = matched_df.apply(lambda x: generate_sentences(x), axis=1)
matched_df = pd.concat([matched_df, new_sentences], axis=1)

matched_df['original_check'] = matched_df.apply(
    lambda x: x.original_sentence == x.active_original, axis=1)
matched_df['updated_check'] = matched_df.apply(
    lambda x: x.updated_sentence == x.active_original, axis=1)
matched_df['matched_check'] = matched_df.apply(
    lambda x: x.matched_sentence == x.passive_original, axis=1)

matched_df.to_csv('test_generated_sentences.csv')

matched_df['source_sent'] = matched_df.index

common_cols = ['original_sentence', 'sub_expt', 'evtype', 'NP1', 'NP2', 'VP',
               'source_sent']

ao_df = matched_df.loc[:, common_cols + ['active_original']]
ar_df = matched_df.loc[:, common_cols + ['active_reversed']]
po_df = matched_df.loc[:, common_cols + ['passive_original']]
pr_df = matched_df.loc[:, common_cols + ['passive_reversed']]

ao_df['voice'] = 'active'
ar_df['voice'] = 'active'
po_df['voice'] = 'passive'
pr_df['voice'] = 'passive'

ao_df['reversal'] = 'original'
ar_df['reversal'] = 'reversed'
po_df['reversal'] = 'original'
pr_df['reversal'] = 'reversed'

dfs = [ao_df, ar_df, po_df, pr_df]
dfs = [df.set_axis(
    common_cols + ['stimulus', 'voice', 'reversal'], axis=1)
    for df in dfs]


stimulus_df = pd.concat(dfs).sort_values(
    by=['source_sent', 'voice', 'reversal'])
stimulus_df = stimulus_df.set_axis(range(len(stimulus_df)), axis=0)
stimulus_df['plausible'] = stimulus_df.apply(
    lambda x: ('implausible' if (x.reversal == 'reversed' and
                                 x.evtype in ['AI', 'AAN']) else 'plausible'),
    axis=1)

# old_data = pd.read_excel("adapt-realmateri als.xls")
# old_sentences = set(old_data["S1"]).union(set(old_data["S2"]))
# stimulus_df['in_old'] = stimulus_df['stimulus'].isin(old_sentences)
# stimulus_df.to_csv('new_sentences.csv')

stimulus_df['index_col'] = stimulus_df.index
stimulus_df['ItemNumber'] = stimulus_df.apply(lambda x: x.index_col // 4 + 1,
                                              axis=1)

df = stimulus_df.loc[:, ['ItemNumber', 'voice', 'plausible', 'stimulus',
                         'original_sentence', 'evtype']]
df = df.set_axis(['ItemNumber', 'Voice', 'Plausible/Implausible', 'Stimulus',
                  'OriginalSentence', 'EvType'], axis=1)
df.to_csv('../newsentences_EventsAdapt.csv')

import pandas as pd
import random
import itertools
from turkolizer import create_turk_file

random.seed(0)


def partition(input_list, n):
    # produces a list on n lists, splitting up input_list into n sublists
    # makes sure each sublist size is a multiple of n, for Latin Square
    # assumes n divides input_list's length
    copy = input_list[:]
    random.shuffle(copy)
    # return [copy[i::n] for i in range(n)]
    q, r = (len(input_list) // n) // n, (len(input_list) // n) % n
    sizes = ([(q + 1) * n] * r) + ([q * n] * (n - r))
    # the sizes of each sublist, makes sure n divides each sublists length
    indices = [sum(sizes[:i]) for i in range(n)] + [len(input_list)]
    # indices to slice up original list at
    sublists = [copy[indices[i]:indices[i + 1]] for i in range(n)]
    random.shuffle(sublists)
    return sublists


def produce_turkolizer_input(df, filename):
    # takes in df, produces txt file for input into turkolizer
    # stores in filename
    with open(filename, 'w') as f:
        # add items
        for idx, row in df.iterrows():
            condition = f"{row['Voice']}-{row['Plausible/Implausible']}"
            # name of condition associated with row
            if row['EvType'] == 'AAR':
                condition += f"-{idx % 2}"
            f.write(f"# {row['EvType']} {row['ItemNumber']} {condition}\n")
            f.write(f"{row['Stimulus']} \n")
            if idx % 4 == 3:
                f.write("\n")

        # add 2 filler questions
        f.write(f"# filler 1 filler \n")
        f.write(f"Please select the leftmost option. \n")
        f.write(f"# filler 2 filler \n")
        f.write(f"Please select the rightmost option. \n")


sent_df = pd.read_csv('newsentences_EventsAdapt.csv', index_col=0)
expt_dict = pd.Series(sent_df.EvType.values,
                      index=sent_df.ItemNumber).to_dict()
# dictionary for item number -> AI/AAN/AAR evtype
expts = ['AI', 'AAN', 'AAR']
parity = {'odd': 1, 'even': 0}
lists_dict = {}

for e in expts:
    for p in parity:
        input_list = [i for i in expt_dict if
                      (expt_dict[i] == e and i % 2 == parity[p])]
        if len(input_list) % 4 != 0:
            input_list = input_list[:- (len(input_list) % 4)]
        # make sure the list is a multiple of 4.
        sublists = partition(input_list, 4)
        # get all the odd or even items expt e, and make 4 sublists

        # very hacky fix below
        if e == 'AAR':
            lists_dict[f"{e}_{p}"] = sorted(sublists, key=len, reverse=False)
        elif e == 'AAN':
            lists_dict[f"{e}_{p}"] = sorted(sublists, key=len, reverse=True)
        elif e == 'AI':
            sublists = sorted(sublists, key=len, reverse=True)
            lists_dict[f"{e}_{p}"] = [sublists[3]] + sublists[0:3]
        # sort sublists so that final lists are equal in length.

# print({k: sum([len(x) for x in v]) for k, v in lists_dict.items()})

for p in parity:
    for i in range(4):
        lists = [lists_dict[f"{e}_{p}"][i] for e in expts]
        chosen_items = list(itertools.chain(*lists))
        # combine one sublist for each evtype to produce lists of ~60 items
        filtered_df = sent_df[sent_df['ItemNumber'].isin(chosen_items)]

        filename = f"turkolizer_files/{p}_{i}.txt"
        produce_turkolizer_input(filtered_df, filename)
        # save the txt file for input into turkolizer

        create_turk_file(filename, 48, 0, 0, FNAME="EventsAdapt2", append=True,
                         code=f"{p}{i}")
        # append to turkolizer csv.

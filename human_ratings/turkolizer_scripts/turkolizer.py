#! env python
# originally written by Kris Fedorenko and Steve Piantadosi in 2009
# many parts rewritten by Richard Futrell in 2015

# modified and commented on by Zawad Chowdhury in 2021 for the computational
# plausibility project at EvLab
from __future__ import print_function
import sys
import random
import codecs
import csv
import copy
import argparse
import functools

try:
    raw_input
except NameError:
    raw_input = input

# object definitions (paradigms for Trial and Experiment)

class Trial(object):
    """
    Takes in a trial string, and procures necessary information from it such as
    self.exp - experiment name
    self.item - item number
    self.condition - trial condition
    self.questions - question, answer pairs (can change read_question_strings)
    self.body - body of trial
    """
    def __init__(self, trial_str):
        self.trial_str = trial_str
        self.lines = trial_str.splitlines()
        self.header = self.lines[0].split()
        self.exp  = self.header[0]
        self.item = self.header[1]
        self.condition = self.header[2].replace("_", "-")
        
        questions = []
        body = []
        for line in self.lines[1:]:
            if line.startswith("?"):
                questions.append(line)
            else:
                body.append(line)

        self.questions = list(read_question_strings(questions))
        self.body = read_body_strings(body)

def read_question_strings(question_strings):
    """ Yield (question, answer) tuples from question strings """
    for question_str in question_strings:
        question_str = question_str.strip("?").strip()
        question, answer = question_str.split("?")
        yield question.strip(), answer.strip()

def read_body_strings(body_strings):
    return "<br/>".join(body_strings)

class Experiment(object):
    """
    Input: trials, dict name -> Trial instances
    Trial name = item + condition
    Procures conditions and items for all the trials in dict
    """
    def __init__(self, trials):
        self.trials = trials
        self.num_trials = len(self.trials)

        conditions = {trial.condition for trial in self.trials.values()}
        self.conditions = sorted(conditions)
        self.num_conditions = len(self.conditions)

        items = {trial.item for trial in self.trials.values()}
        self.items = sorted(items)
        self.num_items = len(self.items)

        for trial in self.trials.values():
            self.num_questions = len(trial.questions)
            # assumes all trial in experiment have the same no of questions
            break

# helper functions

def gcd(a,b):
    """ Return greatest common divisor of a and b. """
    while b:
        a, b = b, a % b
    return a

def lcm(a,b):
    """ Return lowest common multiple of a and b. """
    return int((a*b)/gcd(a,b))

def LCM(terms):
    """" Return lcm of an iterable of numbers. """
    return functools.reduce(lcm, terms)

def prob_choice(lst):
    """
    Input: lst is list of item, weight with weights summing to 1.
    Outputs an item in list, chosen with the appropriate weight.
    Requires Python 3.6
    """
    return random.choices(
        population=[item for (item, weight) in lst],
        weights=[weight for (item, weight) in lst])[0]

def make_header(final):
    """
    Takes the final output, and generates the header of the output csv
    Basically the column headings like "trial__1"
    """
    header = ["list"]
    for i in range(len(final[0])):
        header.append("trial__" + str(i+1))
        num_questions = len(final[0][i].questions)
        for j in range(num_questions):
            header.append("question" + "__" + str(i+1))
        header.append("code__" + str(i + 1))
    return header

def rotate(lst):
    """ Pop the last value of lst and insert it as the first value of lst. """
    value = lst.pop()
    lst.insert(0, value)

def generate_keys(items, conditions):
    conditions = copy.copy(conditions)
    num_conditions = len(conditions)
    ratio = len(items) // num_conditions
    for i in range(ratio):
        for j in range(num_conditions):
            index = j + i * num_conditions
            yield items[index] + conditions[j]

# main functions

def read_file(infile):
    """ Read in a file, returning a dictionary of experiment names to
    experiments. 
    """
    content = infile.read()
    trial_strings = [trial.strip() for trial in content.split("#")]
    trials = [Trial(trial) for trial in trial_strings if trial]
    exp_names = {trial.exp for trial in trials}
    d = {}
    for exp_name in exp_names:
        trial_dict = {trial.item + trial.condition : trial
                      for trial in trials if trial.exp == exp_name}
        d[exp_name] = Experiment(trial_dict)
    return d

def latin_square_lists(d):
    """ Generate Latin Square trial lists for each experiment.
    Return a dict of experiment names to lists of lists of trial objects.
    All item+condition pairs exist, we produce each diagonal of the square for
    each experiment.
    """
    lists = {}
    for exp_name, exp in d.items():
        exp_list = []
        conditions = exp.conditions
        for _ in range(exp.num_conditions):
            L = []
            for j in generate_keys(exp.items, conditions):
                try:
                    L.append(exp.trials[j])  # keys of trial dict are item+cond
                except KeyError:
                    print("Missing trial:")
                    print("        item %s, condition %s" % (j[0], j[1:]))
                    print("        in experiment %s" % exp_name)
                    sys.exit(1)
            exp_list.append(L)
            rotate(conditions)
        lists[exp_name] = exp_list
    return lists

def make_LCM_lists(d, latin_square, lcm):
    """
    creates lcm lists, using each experiment's latin square list lcm/num_cond
    times to produce distinct lists.
    """
    lists = [[] for _ in range(lcm)]
    for exp_name, exp in d.items():
        num = int(lcm / exp.num_conditions)
        for i in range(num):
            for j in range(len(latin_square[exp_name])):
                index = j + i * len(latin_square[exp_name])
                lst = latin_square[exp_name][j]
                lists[index].append(lst[:])
    return lists

def randomize(L, F, Y):
    NEW = []
    ExpNUM = len(L)
    OLD = L[:]
    MinRatio = F

    #        --- creating the first fillers ---
    if Y != 0:
        for i in range(len(OLD)):
            if "filler" in OLD[i][0].exp:
                FILL = i
        FILLERS = []
        for i in range(Y):
            Random_Index = random.choice(range(len(OLD[FILL])))
            FILLERS.append(OLD[FILL][Random_Index])
            OLD[FILL].pop(Random_Index)
    #        ----          ----            ----

    #^*^*^ - calculating initial proportions - ^*^*^
    PROPs = {}
    total = 0
    for exp in OLD:
        total += len(exp)
    #print "total:", total
    for exp in OLD:
        PROPs[exp[0].exp] = len(exp)/float(total)
    #print PROPs
    count = 0
    while True:
        #print "count:", count, "-----------------------------"
        if OLD ==[]:
            break
        exp_to_use = list(range(len(OLD))) # <-- indexes of all exp
        RATIOS = {}
        EXPS = exp_to_use[:]
        for exp in EXPS:             # <-- removes the filler sub expt        from EXPS
            if "filler" in OLD[exp][0].exp:
                EXPS.remove(exp)
        if EXPS == []:
            IsCloseSqueeze = False
        else:
            for i in EXPS:
                expts = exp_to_use[:]
                expts.remove(i)
                len_all = 0
                for index in expts:
                    len_all += len(OLD[index])
                RATIOS[float(len_all)/len(OLD[i])] = i
            SmallestRatio = min(RATIOS.keys())
            IsCloseSqueeze =        MinRatio > SmallestRatio


        #^*^*^ actual placing one of the trials in the NEW list:
        if IsCloseSqueeze:
            trial_to_place = random.choice(OLD[RATIOS[SmallestRatio]])
            NEW.insert(0, trial_to_place)
            OLD[RATIOS[SmallestRatio]].remove(trial_to_place)

        else:
            #^*^*^ remove from the exp_to_use exp that were used last F times
            LastUsed = [] # <-- names of last used exps
            for trial in NEW[:F]:
                expName = trial.exp
                if expName in LastUsed:
                    continue
                LastUsed.append(expName)
            for index in EXPS:
                if OLD[index][0].exp in LastUsed:
                    exp_to_use.remove(index)

            #^*^*^ creates weighted random choice:
            weights = []
            #print "exp_to_use:", exp_to_use

            fillerIndex = None
            for i in exp_to_use:
                Experiment = OLD[i][0].exp
                if "filler" in Experiment:
                    fillerIndex = i
                    continue
                if count< F:
                    k = count
                else:
                    k = F
                #print "k:", k
                weight = round( ((PROPs[Experiment]) / (1 - k * PROPs[Experiment])), 2)
                #print "PROPs[Experiment]:", PROPs[Experiment]
                #print "weight:", weight
                weights.append((i, weight))

            totalW = 0
            for i, w in weights:
                totalW += w
            #print "totalW:", totalW
            if fillerIndex != None:
                weights.append((fillerIndex, round(1-float(totalW), 2)))
            #print "weights:", weights


            randIndex = prob_choice(weights) # weighted
            trial_to_place = random.choice(OLD[randIndex])
            NEW.insert(0, trial_to_place)
            OLD[randIndex].remove(trial_to_place)

        #^*^*^ removes empty exp lists
        try:
            OLD.remove([])
        except ValueError:
            pass

        count += 1
    # ---- Adding first fillers ----
    if Y != 0:
        for f in FILLERS:
            NEW.insert(0, f)

    return NEW

def finalize(lsts, F, Y):
    """ Randomize each experiment list, and from them create final versions of
    the lists of the Trial objects. """
    for lst in lsts:
        for x in lst:
            random.shuffle(x)
    return [randomize(lst, F, Y) for lst in lsts]

def check(d, F, Y, N, lcm):
    # number of trials should be a multiple of number of conditions
    for exp_name, exp in d.items():
        if exp.num_trials != exp.num_conditions * exp.num_items:
            print("Error in the text file:")
            print("        trials are missing for some of the conditions")
            print("        in the experiment %s" % exp_name)
            print("        (might be something wrong with the headings of the trials)")

        # ^*^ number of items should be a multiple of number of conditions: ^*^
        if float(exp.num_items) % exp.num_conditions != 0:
            print("Error in the text file:")
            print("        Number of items is not a multiple of number of conditions")
            print("        in the experiment %s" % exp_name)
            sys.exit(1)

    # input number of lists should be a multiple of LCM of conditions
    if float(N) % lcm != 0:
        print("Input Error:")
        print("         number of lists requested (%d) is not a multiple of" % N)
        print("         LCM of conditions (%d)" % lcm)
        sys.exit(1)

    # checking F (number of in-between trials):
    nonfiller_exp_names = {name for name in d if "filler" not in name}
    no_fillers = len(nonfiller_exp_names) == len(d)
    if no_fillers:
        print("WARNING:")
        print("            fillers are missing or there are problems")
        print("            with their format in the text file.")
        print("                    (e.g. the experiment title of fillers")
        print("                must include the word 'filler')")
        
    for exp_name in nonfiller_exp_names:
        num_other_items = sum(exp.num_items for name, exp in d.items()
                              if name != exp_name)
        # num_other_items always includes fillers and first Y fillers are not
        # used in the algorithm with F, so we should look at num_other_items - Y
        if num_other_items - Y < F * d[exp_name].num_items:
            print("WARNING:")
            print("        there might not be enough items to have %d" % F)
            print("        trials from other experiments and fillers")
            print("        between trials of %s experiment." % exp_name)
            print("        please choose a smaller value for the number of")
            print("        in-between trials / decrese the number of")
            print("            fillers in the beginning of each list")
            print("        or add more fillers.")


    # checking the number of questions (should be the same for all the experiments):
    question_numbers = [exp.num_questions for exp in d.values()]
    for i in range(len(question_numbers)):
        if question_numbers[i] != question_numbers[i-1]:
            print("Error in the text file:")
            print("        the number of questions in trials")
            print("        in different experiments")
            print("        is not the same")
            sys.exit(1)

def make_rows(list_of_trials):
    for i, trials in enumerate(list_of_trials):
        newrow = []
        newrow.append(str(i+1))
        for trial in trials:
            newrow.append(trial.body)
            answer = "NO_QUESTION"
            try:
                for question, answer in trial.questions:
                    newrow.append(question+'?')
            except ValueError:
                print("Error in the text file:")
                print(" there is something wrong with question")
                print(" and/or answer in the trial of experiment %s," % trial.exp)
                print(" item: %s, condition: %s" % (trial.item, trial.condition))
                sys.exit(1)
            newrow.append("_".join([trial.exp,
                                    trial.condition,
                                    trial.item,
                                    answer]))
        yield newrow


def create_turk_file(filename=None, N=None, F=None, Y=None, seed=None,
                     FNAME=None, append=False):
    if filename is None:  # two modes, either pass in all args or get input
        filename, N, F, Y, seed = get_args()
    if seed is not None:
        random.seed(seed)

    if FNAME is None:
        FNAME = filename.split("/")[-1]
        FNAME = FNAME.split(".")[0]

    print()
    print("Processing the text file...")
    print()
    print("-------")
    with codecs.open(filename, encoding="utf-8") as infile:
        d = read_file(infile)
    print("Number of experiments: %d" % len(d))
    for exp_name, exp in d.items():
        print()
        print("Experiment: %s" % exp_name)
        print("          - %d items" % exp.num_items)
        print("          - %d conditions" % exp.num_conditions)
        print("          - %d trials" % exp.num_trials)
        print("          - number of questions: %d" % exp.num_questions)
        print("          - conditions: %s" % exp.conditions)
    print("-------")
    print()

    cond_nums = [exp.num_conditions for exp in d.values()]
    lcm = LCM(cond_nums)

    print("Performing a check of the parameters...")
    check(d, F, Y, N, lcm)

    print()
    print("Creating a latin square...")
    # creates latin-square list of lists for all experiments
    # (each experiment is a list of n lists, where n is number of conditions)
    # 3 levels embedded lists
    latin_square = latin_square_lists(d)

    print()
    print("Creating LCM (%d) lists..." % lcm)
    LCM_lists = make_LCM_lists(d, latin_square, lcm)

    print()
    print("Creating %d lists..." % N)
    N_lists = [copy.deepcopy(lst) for _ in range(int(N / lcm)) for lst in LCM_lists]

    print()
    print("Randomizing each list...")
    final = finalize(N_lists, F, Y)

    #********************* TO CHECK THE RANDOMIZING: *****************
    #for newlist in FINAL:
    #        expments = []
    #        [expments.append(trial.exp) for trial in newlist]
    #        print "-------", expments

    print()
    print("Creating csv file...")
    header = make_header(final)
    rows = make_rows(final)

    mode = 'ab' if append else 'wb'
    with codecs.open(FNAME + '.turk.csv', mode, encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print()
    print("------- DONE! -------")
    print("Result saved in %s" % FNAME + ".turk.csv")
    print()

def get_args():
    args = parse_args()
    filename = args.filename if args.filename else raw_input("Please enter the name of the text file: ")
    N = args.num_lists if args.num_lists else raw_input("Please enter the desired number of lists: ")
    F = args.num_inbetween if args.num_inbetween else raw_input("Please enter the desired number of in-between trials: ")
    Y = args.num_beginning_fillers if args.num_beginning_fillers else raw_input("Please enter the desired number of fillers in the beginning of each list: ")
    seed = args.seed
    return filename, int(N), int(F), int(Y), seed
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Latin Square experiment lists for Mechanical Turk from Linger-formatted stimuli.",
        epilog="Example usage: python turkolizer.py example_turk_format.txt 32 1 1",
        )
    parser.add_argument('filename', type=str, help="Filename of Linger-formatted stimuli.", nargs='?', default=None)
    parser.add_argument('num_lists', type=int, help="Desired number of lists.", nargs='?', default=None)
    parser.add_argument('num_inbetween', type=int, help="Desired number of in-between trials.", nargs='?', default=None)
    parser.add_argument('num_beginning_fillers', type=int, help="Desired number of fillers in the beginning of each list.", nargs='?', default=None)
    parser.add_argument('-s', '--seed', type=int, help="Seed for pseudorandom numbers.", nargs='?', default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    create_turk_file()

import pandas as pd
from turkolizer import create_turk_file


def get_active_items_dicts():
    df = pd.read_csv('sentences_EventsAdapt.csv')
    items = (
        {'AI': [], 'AAN': [], 'AAR': []},  # even
        {'AI': [], 'AAN': [], 'AAR': []},  # odd
    )
    active = df[df['Voice'] == 'Active']

    for idx, row in active.iterrows():
        condition = f'{row["Voice"]}-{row["Plausible/Implausible"]}'
        ev_type = row['EvType']
        lst = items[int(row['ItemNumber']) % 2][ev_type]
        if ev_type == 'AAR':
            condition += f'-{idx % 2 + 1}'
        if idx % 4 == 0:
            lst.append('')
        lst[-1] += f'# {row["EvType"]} {row["ItemNumber"]} {condition}\n' \
                   f'{row["Stimulus"]}\n'

    return items


def add_items_to_file(csv, items, trials_in_exp_num):
    idx = {'AI': 0, 'AAN': 0, 'AAR': 0}
    for _ in range(40):
        with open('output.txt', 'w') as txt:
            for exp in 'AI', 'AAN', 'AAR':
                for _ in range(trials_in_exp_num):
                    txt.write(items[exp][idx[exp]])
                    idx[exp] += 1
                    idx[exp] %= len(items[exp])
        create_turk_file('output.txt', 64, 0, 0)
        # add result to csv


def main():
    even, odd = get_active_items_dicts()
    with open('EventsAdapt.turk.csv', 'w') as csv:
        add_items_to_file(csv, even, trials_in_exp_num=64)
        add_items_to_file(csv, odd, trials_in_exp_num=64)


if __name__ == '__main__':
    main()

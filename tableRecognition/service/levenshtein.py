recnik = [
    'od kojih zasićene masne kiseline',
    'prosečne hranljive vrednosti na 100g proizvoda',
    'prosečne nutritivne vrednosti na 100g proizvoda'
    'energetska vrednost',
    'energija',
    'proteini',
    'so',
    'vlakna',
    'od kojih šećeri',
    'masti',
    'ugljeni hidrati'
]


def levenshtein_(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_real_word(word):
    word = word.lower()
    best = 100
    idx_dictonary = -1
    for i, rec in enumerate(recnik):
        res = levenshtein_(word, rec)
        if (res < best):
            best = res
            idx_dictonary = i
    return recnik[idx_dictonary]


def handle_number(s1):
    for i, char in enumerate(s1):
        if not char.isdigit() and char not in ['J', 'k', '/']:
            if char == 'o' or char == 'O':
                s1 = s1[:i] + '0' + s1[i + 1:]
            elif char == 's' or char == 'S':
                s1 = s1[:i] + '5' + s1[i + 1:]
            elif char == '9' or char == 'q' or char == 'g':
                s1 = s1[:i] + 'g' + s1[i + 1:]
            elif char == ' ':
                continue
            else:
                s1 = s1[:i] + ',' + s1[i + 1:]
    s1 = s1[:(len(s1) - 4)] + 'kcal'
    return s1


def get_number(s1):
    if len(s1) > 10:
        return handle_number(s1)
    for i, char in enumerate(s1):
        if not char.isdigit():
            if char == 'o' or char == 'O':
                s1 = s1[:i] + '0' + s1[i + 1:]
            elif char == 's' or char == 'S':
                s1 = s1[:i] + '5' + s1[i + 1:]
            elif (char == 'q' or char == 'g') and i != (len(s1)-1):
                s1 = s1[:i] + '9' + s1[i + 1:]
            elif char == '%':
                continue
            else:
                s1 = s1[:i] + ',' + s1[i + 1:]
    return s1

def digit_count(text):
    cnt = 0
    for c in text:
        if c.isdigit():
            cnt = cnt + 1
    return cnt

def convert_to_real_word(text):
    if (text[0].isdigit()):
        return get_number(text)
    elif text[0] == 'o' or text[0] == 'O':
        if 2 <= len(text) <= 6 and digit_count(text) > 0:
            return get_number(text)
        else:
            return get_real_word(text)
    else:
        return get_real_word(text)

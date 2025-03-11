word_sep = u" ,.?:;'\"/<>`!$%^&*()-=+~[]\\|{}()\n\t" \
            + u"©℗®℠™،、⟨⟩‒–—―…„“”–――»«›‹‘’：（）！？=【】　・" \
            + u"⁄·† ‡°″¡¿÷№ºª‰¶′″‴§|‖¦⁂❧☞‽⸮◊※⁀「」﹁﹂『』﹃﹄《》―—" \
            + u"“”‘’、，一。►…¿«「」ー⋘▕▕▔▏┈⋙一ー।;!؟"
word_sep = u'#' + word_sep
translate_table = dict((ord(char), u' ') for char in word_sep)

def is_emoji(input):
    if not input:
        return False
    if u"\U0001F600" <= input and input <= u"\U0001F64F":
        return True
    elif u"\U0001F300" <= input and input <= u"\U0001F5FF":
        return True
    elif u"\U0001F680" <= input and input <= u"\U0001F6FF":
        return True
    elif u"\U0001F1E0" <= input and input <= u"\U0001F1FF":
        return True
    elif u"\U0001F9FF" == input:
        return True
    else:
        return False
    
def remove_emoji(input):
    res = ''
    for i in input:
        if is_emoji(i):
            continue
        else:
            res += i
    return res

def clean_word_bag(text, stopwords=[], keyword=[]):
    # get rid of URL
    original_text = str(text).lower()
    tok = original_text.split(' ')
    text = u''
    for x in tok:
        x = remove_emoji(x)
        if len(keyword) > 0:
            if x not in keyword: continue
        elif len(stopwords) > 0:
            if len(x) == 0:
                continue
            elif x[0:4] == 'http' or x[0:5] == 'https':
                continue
            elif x[0] == '@':
                continue
            elif x[0] == '&':
              continue
            elif x.isdigit():
                continue
            elif x in stopwords:
                continue
        for i in range(len(x)-1):
            if x[i] == '@':
                x = x[:i]
                break
        text = text + ' ' + x
    tokens = text.translate(translate_table).split(' ')
    new_tokens = []
    for i in tokens:
        if i in stopwords:
            continue
        else:
            new_tokens.append(i)
    return sorted(list(filter(lambda word: len(word) >= 2, new_tokens)))

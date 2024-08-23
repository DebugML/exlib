import torch

## word, phrase, and sentence-level chunks

def text_chunk(word_list, baseline='word'):
    assert baseline in ['word', 'phrase', 'sentence']
    
    groups = []
    if baseline == 'word':
        for word in word_list:
            groups.append([word])
    elif baseline == 'phrase':
        #each group is 3 consecutive words
        for i in range(0, len(word_list), 3):
            groups.append(word_list[i:i+3])
    elif baseline == 'sentence':
        #reconstruct sentences from word list
        sentence = ""
        for word in word_list:
            sentence += word + " "
            if word[-1] == "." or word[-1] == "!" or word[-1] == "?":
                groups.append(sentence.split())
                sentence = ""
        if(len(sentence) > 0):
            groups.append(sentence.split())
    return groups
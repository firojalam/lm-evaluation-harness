import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np

def bleu_score(items):
    nltk.download('wordnet')
    unzipped_list = list(zip(*items))
    references = unzipped_list[0]
    candidates = unzipped_list[1]
    bleu_scr = corpus_bleu([[r] for r in references], candidates)
    return bleu_scr

def meteor_score(items):
    # print(items)
    unzipped_list = list(zip(*items))
    references = unzipped_list[0]
    candidates = unzipped_list[1]
    meteor_score_sentences_list=[]
    def corpus_meteor(predicted, references):
        meteor_score_sentences_list = list()
        for reference, predict in zip(references, predicted):
            try:
                meteor_score_sentences_list.append(meteor_score([word_tokenize(reference)], word_tokenize(predict)))
            except:
                pass
    meteor_score_res = np.mean(meteor_score_sentences_list)
    return meteor_score_res

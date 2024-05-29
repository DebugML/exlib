import torch
from tqdm import tqdm
import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, sent_tokenize
from .common import *
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs', 'LAL-Parser'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs', 'LAL-Parser', 'src_joint'))
# import importlib  
# lal_parser = importlib.import_module(".libs.LAL-Parser", package="exlib.explainers")
# KM_parser = importlib.import_module(".libs.LAL-Parser.src_joint.KM_parser", package="exlib.explainers")
# REVERSE_TOKEN_MAPPING = importlib.import_module(".libs.LAL-Parser.src_joint.main.REVERSE_TOKEN_MAPPING", package="exlib.explainers")
# torch_load_parser = importlib.import_module(".libs.LAL-Parser.src_joint.main.torch_load", package="exlib.explainers")
from src_joint import KM_parser
from src_joint.main import REVERSE_TOKEN_MAPPING
from src_joint.main import torch_load as torch_load_parser
from .libs.idg.utils import load_model as load_model_idg
from .libs.idg.calculate_gradients import execute_IDG


class IDGTextCls(FeatureAttrMethod):
    def __init__(self, model, tokenizer, parser=None, bert=True):
        super().__init__(model)

        self.tokenizer = tokenizer
        if parser is None:
            curr_file_path = os.path.dirname(os.path.abspath(__file__))
            parser_path = os.path.join(curr_file_path, 'libs', 'LAL-Parser', 'best_parser.pt')
            info = torch_load_parser(parser_path)
            parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
            parser.eval()
        self.parser = parser
        self.bert = bert # can only do bert or xlnet for now

    def forward(self, x, t=None):
        """
        X str: list of sentences
        """
        tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(tag, tag), REVERSE_TOKEN_MAPPING.get(word, word)) \
                     for word, tag in nltk.pos_tag(word_tokenize(sentence))] for sentence in x]
        
        syntree, _ = self.parser.parse_batch(tagged_sentences)
        trees = [syntree[i].convert().linearize() for i in range(len(syntree))]

        coalitions_all = []
        value_func_all = []
        single_attr_all = []
        for i, tree in enumerate(trees):
            coalitions_i, value_func_i = execute_IDG([trees[i]], self.model, self.tokenizer, t[i], bert=self.bert)
            coalitions_all.append(coalitions_i)
            value_func_all.append(value_func_i)

        return FeatureAttrOutput(torch.tensor(value_func_all), {'coalitions': coalitions_all, 'value_func': value_func_all})
import torch
import torch.nn as nn 
from bertopic import BERTopic
from tqdm import tqdm 

class ClusteringGroups(nn.Module):
    def __init__(
        self, 
        utterances,
        distinct: int,
        scaling = 1.5,
        topic_model_language = 'multilingual'
    ):
        super().__init__()
        self.scaling = scaling
        self.distinct = distinct
        self.max_groups = int(scaling * distinct)
        self.utterances = utterances
        self.topic_model = BERTopic(language=topic_model_language) 
        self.topics, self.probs = self.topic_model.fit_transform(utterances)


    def forward(self, x): # assumes that x is a word_list (so it is an utterance.split())
        masks = []
        # each feature group/mask should represent a topic
        for topic in self.topic_model.get_topic_info()['Representation']:
            mask = torch.zeros(len(x))
            topic_represented = False
            for word_i in range(len(topic)):
                if topic[word_i] in x:
                    mask[word_i] = 1
                    topic_represented = True
            if topic_represented and len(masks) <= self.max_groups:
                masks.append(mask)
        return masks
    
from exlib.explainers.archipelago import ArchipelagoTextCls
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return outputs.logits
    
class ArchipelagoGroups(nn.Module):
    def __init__(
        self, 
        backbone_model,
        distinct: int,
        scaling = 1.5
    ):
        super().__init__()
        self.scaling = scaling
        self.distinct = distinct
        self.max_groups = int(scaling * distinct)
        self.model = WrappedModel(backbone_model).to(device)
        self.explainer = ArchipelagoTextCls(backbone_model).to(device)
    
    
    def forward(self, batch):
        if not isinstance(batch['input_ids'], torch.Tensor):
            inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
            if 'token_type_ids' in batch:
                token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
            else:
                token_type_ids = None
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

            # concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
            # segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()
            # print('segs', segs.shape)
        else:
            inputs = batch['input_ids'].to(device)
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = None
            attention_mask = batch['attention_mask'].to(device)
            # segs = batch['segs'].to(device).float()

#         labels = batch['label'].to(device)
        inputs_dict = {
            'input_ids': inputs,
#             'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        
#         logits = self.model(**inputs_dict).logits
        logits = self.model(**inputs_dict)

        
        preds = torch.argmax(logits, dim=-1)
        kwargs = {
#             'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }
        expln = self.explainer(inputs, preds, **kwargs)
        
        word_lists = batch['word_list']
        word_lists = list(map(list, zip(*word_lists)))
        processed_word_lists = []
        for word_list in word_lists:
            processed_word_lists.append([word for word in word_list if word != ''])
        
        all_batch_masks = []
        for i in range(len(processed_word_lists)):
            ngroups = min(self.max_groups, len(processed_word_lists[i]))
            all_groups_pos = torch.topk(expln.attributions[i][:len(processed_word_lists[i])], len(processed_word_lists[i])).indices
            all_groups_pos = torch.tensor_split(all_groups_pos, ngroups)
            masks = []
            for groups_pos in all_groups_pos:
                mask = torch.zeros(len(word_lists[i]))
                for pos in groups_pos:
                    mask[pos] = 1
                masks.append(mask)
            all_batch_masks.append(masks)
        return all_batch_masks
from transformers import (
    BertPreTrainedModel, 
    BertModel, 
    RobertaPreTrainedModel,
    RobertaModel,
) 
from transformers.modeling_outputs import TokenClassifierOutput
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import faiss
from faiss import normalize_L2
class BertNegSampleForTokenClassification(BertPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)
        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


class RobertaNegSampleForTokenClassification(RobertaPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)



        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
        

class RobertaNegSampleForTokenClassificationCL(RobertaPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):

        hidden_states = self.get_hidden_states(input_ids,attention_mask,start_pos)

        hidden_states, labels_mat = self.reshape(hidden_states, labels_mat)

        logits = self.cls(hidden_states)

        loss_cl = self.cl(hidden_states)

        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return loss, loss_cl

    def inference(self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
        ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)

        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
    def get_hidden_states(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)
        return hidden_states

    def reshape(self, hidden_states, labels):
        indexes = torch.where(labels != -100)
        labels = labels[indexes]
        hidden_states = hidden_states[indexes]
        return hidden_states, labels

    def cl(self, y_pred):
        device = y_pred.device
        n = y_pred.shape[0]
        y_true = torch.arange(y_pred.shape[0], device=device)
        y_true = (y_true + n/2) % n 
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / 0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        y_true = y_true.long()
        loss = F.cross_entropy(sim, y_true)
        return loss

class RobertaNegSampleForTokenClassificationCLSel(RobertaPreTrainedModel):
    
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config, mlp_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooling = nn.Sequential(
            nn.Linear(config.hidden_size * 4, mlp_config["hidden_size"]),
            nn.Tanh(),            
        )
        self.cls = nn.Sequential(
            nn.Dropout(mlp_config["dropout_rate"]),
            nn.Linear(mlp_config["hidden_size"], self.num_labels),
        )
        index = faiss.IndexFlatIP(256)  # 使用 L2 距离进行索引
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)  # 将索引移动到 GPU 上
        self.index = index
        self.post_init()
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
    ):

        hidden_states = self.get_hidden_states(input_ids,attention_mask,start_pos)
        
        hidden_states, labels_mat = self.reshape(hidden_states, labels_mat)

        logits = self.cls(hidden_states)

        loss_cl = self.cl(hidden_states)

        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return loss, loss_cl

    def get_neg_mat(self,
        pos_mat:Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None,
        ):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)

        pos_index = torch.where(pos_mat == 1)
        neg_index = torch.where(pos_mat.view(-1) == 0)

        # import pdb
        # pdb.set_trace()

        query = hidden_states[pos_index]
        batch_size, n, _, hidden_size = hidden_states.shape

        hidden_states = hidden_states.view(batch_size * n * n, hidden_size)

        neg_map = torch.zeros(batch_size * n * n,dtype=int)
        neg_map[neg_index] = 1

        query = F.normalize(query)
        hidden_states = F.normalize(hidden_states)

        # import pdb
        # pdb.set_trace()

        self.index.add(hidden_states.cpu().detach().numpy())
        distances, indices = self.index.search(query.cpu().detach().numpy(), 5)
        self.index.reset()

        for i in range(indices.shape[0]):
            neg_map[indices[i]] = 0
        
        return neg_map
        



    def inference(self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        labels_mat: Optional[torch.Tensor] = None
        ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)

        logits = self.cls(hidden_states)
        
        loss_func = CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.num_labels), labels_mat.view(-1)) if labels_mat is not None else None
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
    
    def get_hidden_states(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        positions = start_pos.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        sequence_output = torch.gather(sequence_output, dim=-2, index=positions)
        token_num = sequence_output.shape[1]
        ext_row = sequence_output.unsqueeze(2).expand(-1, token_num, token_num, -1)
        ext_col = sequence_output.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_col, ext_row - ext_col, ext_row * ext_col], dim=-1)
        hidden_states = self.pooling(table)
        return hidden_states

    def reshape(self, hidden_states, labels):
        indexes = torch.where(labels != -100)
        labels = labels[indexes]
        hidden_states = hidden_states[indexes]
        return hidden_states, labels

    def cl(self, y_pred):
        device = y_pred.device
        n = y_pred.shape[0]
        y_true = torch.arange(y_pred.shape[0], device=device)
        y_true = (y_true + n/2) % n 
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / 0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        y_true = y_true.long()
        loss = F.cross_entropy(sim, y_true)
        return loss
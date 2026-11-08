"""Lightweight standalone smoke tests; installs temporary RecBole stubs only.

Run: python test_proposedsis.py
"""

import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import importlib.util

# In this sandbox the actual recbole packages are not installed.
recbole = types.ModuleType('recbole')
recbole_model = types.ModuleType('recbole.model')
recbole_loss = types.ModuleType('recbole.model.loss')
recbole_utils = types.ModuleType('recbole.utils')
class BPRLoss(nn.Module):
    def forward(self, pos, neg):
        return F.softplus(-(pos-neg)).mean()
recbole_loss.BPRLoss = BPRLoss
recbole_utils.InputType = types.SimpleNamespace(PAIRWISE='PAIRWISE')
for name, mod in [('recbole', recbole), ('recbole.model', recbole_model),
                  ('recbole.model.loss', recbole_loss), ('recbole.utils', recbole_utils)]:
    sys.modules[name] = mod

class CrossDomainRecommender(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.total_num_users = dataset.n_users
        self.total_num_items = dataset.n_items
        self.overlapped_num_users = dataset.num_overlap_user
        self.overlapped_num_items = dataset.num_overlap_item
        self.target_num_users = dataset.n_users
        self.target_num_items = dataset.n_items
        self.SOURCE_USER_ID = 'SU'
        self.SOURCE_ITEM_ID = 'SP'
        self.SOURCE_NEG_ITEM_ID = 'SN'
        self.TARGET_USER_ID = 'TU'
        self.TARGET_ITEM_ID = 'TP'
        self.TARGET_NEG_ITEM_ID = 'TN'

cdr = types.ModuleType('recbole_cdr')
model = types.ModuleType('recbole_cdr.model')
parent = types.ModuleType('recbole_cdr.model.crossdomain_recommender')
parent.CrossDomainRecommender = CrossDomainRecommender
for name, mod in [('recbole_cdr', cdr), ('recbole_cdr.model', model),
                  ('recbole_cdr.model.crossdomain_recommender',parent)]:
    sys.modules[name] = mod

# Load the real uploaded baseline under its expected import path.
base_spec = importlib.util.spec_from_file_location(
    'recbole_cdr.model.cross_domain_recommender.dgcdrbase',
    Path(__file__).parent / 'recbole_cdr/model/cross_domain_recommender/dgcdrbase.py')
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)

sis_spec = importlib.util.spec_from_file_location('proposed_sis',
    Path(__file__).parent / 'recbole_cdr/model/cross_domain_recommender/proposedsis.py')
sis = importlib.util.module_from_spec(sis_spec)
sis_spec.loader.exec_module(sis)

class Dataset:
    n_users = 6
    n_items = 9
    num_overlap_user = 3
    num_overlap_item = 1
    source_user_ID_remap_dict = {'[PAD]':0,'a':1,'b':2,'source':3}
    target_user_ID_remap_dict = {'[PAD]':0,'a':1,'b':2,'target':4}
    def inter_matrix(self, form, value_field, domain):
        assert form == 'coo'
        if domain == 'source':
            rows = [1,1,1,2,3,3,3]
            cols = [1,2,3,2,1,2,4]
        else:
            rows = [1,1,2,2,4,4,4]
            cols = [1,5,1,6,7,7,8]
        return sp.coo_matrix((np.ones(len(rows)), (rows,cols)),
                             shape=(self.n_users,self.n_items))

config = {
    'embedding_size':8, 'n_layers':1, 'drop_rate':0.0,
    'connect_way':'mean','full_sort_chunk_size':3,
    'preference_dim':4,'pretrained_model_file':'/tmp/nonexistent.ckpt',
    'lambda_preference_rank':1.0,'lambda_adv':0.1,
    'lambda_cross':0.1,'adv_sample_size':4,'cross_sample_size':4,
    'source_domain':{'dataset':'source-test'},
    'target_domain':{'dataset':'target-test'},
}
data = Dataset()
backbone = base.DGCDRBASE(config, data)
checkpoint = {'config': {'model':'DGCDRBASE', 'embedding_size':8,'n_layers':1,
              'connect_way':'mean','source_domain':{'dataset':'source-test'},
              'target_domain':{'dataset':'target-test'}},
              'state_dict':backbone.state_dict()}

m = sis.PROPOSEDSIS(config, data)
status = m.load_pretrained_backbone_checkpoint(checkpoint)
assert status['cache_ready']
assert not m.source_user_embedding.weight.requires_grad
assert m.source_user_preference_projection.weight.shape == (4,8)

batch = {
    'SU':torch.tensor([1,2,3]),'SP':torch.tensor([1,2,4]),'SN':torch.tensor([4,1,3]),
    'TU':torch.tensor([1,2,4]),'TP':torch.tensor([1,6,7]),'TN':torch.tensor([6,8,1]),
}
torch.manual_seed(1)
parts = m.calculate_loss_components(batch)
for name, val in parts.items():
    assert torch.isfinite(val).all(),(name,val)
assert parts['adversarial_classifier'].item() > 0
sum(m.calculate_loss(batch)).backward()
assert m.target_item_preference_projection.weight.grad is not None
assert m.target_user_popularity_coefficient.weight.grad is not None
assert m.target_popularity_classifier[0].weight.grad is not None
assert m.source_user_embedding.weight.grad is None
assert m.full_sort_predict({'TU':torch.tensor([1,4])}).shape == (18,)
assert m.predict({'TU':torch.tensor([1,4]),'TP':torch.tensor([1,7])}).shape == (2,)
assert (m.source_item_preference_projection.weight.grad is not None)

m.zero_grad(set_to_none=True)
p = m._preference_representations()
m._cross_loss(p).backward()
assert m.source_user_preference_projection.weight.grad is None
assert m.target_user_preference_projection.weight.grad is not None

x = torch.tensor([1.0], requires_grad=True)
(sis.gradient_reverse(x, 2.0)*3).sum().backward()
assert torch.allclose(x.grad, torch.tensor([-6.0]))

saved = m.state_dict()
second = sis.PROPOSEDSIS(config, data)
second.load_state_dict(saved)
assert second._frozen_cache_ready
assert torch.allclose(m.source_z_user, second.source_z_user)

for adv, cross in [(0.0,0.1),(0.1,0.0)]:
    cfg = {**config,'lambda_adv':adv,'lambda_cross':cross}
    ablation = sis.PROPOSEDSIS(cfg,data)
    ablation.load_pretrained_backbone_checkpoint(checkpoint)
    components = ablation.calculate_loss_components(batch)
    assert torch.isfinite(components['total'])
    if adv == 0:
        assert components['adversarial_classifier'].item() == 0
    if cross == 0:
        assert components['cross'].item() == 0
print('PASS: M2/M3/M4, checkpoint, BPR, GRL sign, one-way cross, backward, predict, restore')

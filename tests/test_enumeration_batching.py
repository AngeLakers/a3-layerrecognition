"""Sample isolation across mixed-K path batches and their gradients."""
import pytest
import torch
from a3_ultrasound.inference.enumerator import enumerate_binary_chain
from a3_ultrasound.training import ModelConfig,build_model


@pytest.mark.parametrize('head',['local_independent','local_pre_independent','vanilla_chain','proposed'])
def test_grouped_enumeration_matches_samplewise_values_and_gradients(head):
    torch.manual_seed(412)
    model=build_model(ModelConfig(backbone='tcn',head_type=head,query_dim=64,token_dim=64,attention_dim=64)).eval()
    counts=torch.tensor([6,1,3,2,5,4,1,6,2,4,3,5])
    mask=torch.arange(6)[None]<counts[:,None]
    inputs=[torch.randn(12,6,64,requires_grad=True) for _ in range(4)]
    actual=enumerate_binary_chain(model.head,*inputs,mask)
    expected=[enumerate_binary_chain(model.head,*(x[i:i+1] for x in inputs),mask[i:i+1]) for i in range(12)]
    for i,k in enumerate(counts.tolist()):
        n=2**k
        for field in ('probabilities','log_probabilities','raw_log_probabilities'):
            torch.testing.assert_close(getattr(actual,field)[i,:n],getattr(expected[i],field)[0,:n],atol=1e-6,rtol=1e-5)
        for field in ('marginals','conditional_probability_range','map_states'):
            torch.testing.assert_close(getattr(actual,field)[i,:k],getattr(expected[i],field)[0,:k],atol=1e-6,rtol=1e-5)
        assert not actual.path_mask[i,n:].any()
        assert (actual.map_states[i,k:]==-1).all()
    parameters=inputs+list(model.head.parameters())
    grad_a=torch.autograd.grad(-actual.log_probabilities[:,0].sum(),parameters,allow_unused=True)
    grad_b=torch.autograd.grad(-sum(x.log_probabilities[0,0] for x in expected),parameters,allow_unused=True)
    for a,b in zip(grad_a,grad_b):
        if a is None or b is None:assert a is b
        else:torch.testing.assert_close(a,b,atol=3e-6,rtol=2e-5)

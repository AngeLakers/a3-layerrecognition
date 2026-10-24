"""Check the declared physical Gaussian bias, independent of content scores."""
import pytest
import torch

from a3_ultrasound.models.indexed_readout import IndexedReadout, build_indexed_supports


@pytest.mark.parametrize('sigma,strength', [(36.,0.),(36.,.5),(36.,1.),(50.,.5)])
def test_prior_matches_declared_attention_equation(sigma,strength):
    centers=torch.arange(301,dtype=torch.float32)
    mask=torch.ones(1,1,dtype=torch.bool)
    local,_,prior=build_indexed_supports(centers,torch.tensor([[80.]]),
        torch.tensor([[220.]]),torch.ones(1,301,dtype=torch.bool),mask,
        prior_center=torch.tensor([[150.]]),prior_sigma_samples=sigma)
    readout=IndexedReadout(1,1,1,prior_strength=strength)
    with torch.no_grad():
        for parameter in readout.parameters():parameter.zero_()
    out=readout(torch.ones(1,301,1),torch.ones(1,1,1),local,mask,local_prior=prior)
    expected=torch.softmax(-strength*(centers[80:221]-150).square()/(2*sigma**2),dim=0)
    torch.testing.assert_close(out.attention[0,0,80:221],expected)
    assert torch.count_nonzero(out.attention[~local])==0
    torch.testing.assert_close(out.token,torch.ones(1,1,1))


@pytest.mark.parametrize('strength',[-1.,float('nan'),float('inf')])
def test_invalid_prior_strength_rejected(strength):
    with pytest.raises(ValueError):IndexedReadout(1,1,1,prior_strength=strength)

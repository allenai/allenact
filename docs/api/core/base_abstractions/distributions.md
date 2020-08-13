# core.base_abstractions.distributions [[source]](https://github.com/allenai/allenact/tree/master/core/base_abstractions/distributions.py)

## AddBias
```python
AddBias(self, bias: torch.FloatTensor)
```
Adding bias parameters to input values.
### forward
```python
AddBias.forward(self, x: torch.FloatTensor) -> torch.FloatTensor
```
Adds the stored bias parameters to `x`.
## Bernoulli
```python
Bernoulli(self, num_inputs, num_outputs)
```
A learned Bernoulli distribution.
## CategoricalDistr
```python
CategoricalDistr(self, probs=None, logits=None, validate_args=None)
```
A categorical distribution extending PyTorch's Categorical.
## DiagGaussian
```python
DiagGaussian(self, num_inputs, num_outputs)
```
A learned diagonal Gaussian distribution.
## FixedBernoulli
```python
FixedBernoulli(self, probs=None, logits=None, validate_args=None)
```
A fixed Bernoulli distribution extending PyTorch's Bernoulli.
## FixedNormal
```python
FixedNormal(self, loc, scale, validate_args=None)
```
A fixed normal distribution extending PyTorch's Normal.

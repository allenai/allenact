# core.algorithms.onpolicy_sync.losses.a2cacktr [[source]](https://github.com/allenai/allenact/tree/master/core/algorithms/onpolicy_sync/losses/a2cacktr.py)
Implementation of A2C and ACKTR losses.
## A2C
```python
A2C(self, value_loss_coef, entropy_coef, *args, **kwargs)
```
A2C Loss.
## A2CACKTR
```python
A2CACKTR(self, value_loss_coef, entropy_coef, acktr=False, *args, **kwargs)
```
Class implementing A2C and ACKTR losses.

__Attributes__


- `acktr `: `True` if should use ACKTR loss (currently not supported), otherwise uses A2C loss.
- `value_loss_coef `: Weight of value loss.
- `entropy_coef `: Weight of entropy (encouraging) loss.

## ACKTR
```python
ACKTR(self, value_loss_coef, entropy_coef, *args, **kwargs)
```
ACKTR Loss.

This code is not supported as it currently lacks an implementation
for recurrent models.


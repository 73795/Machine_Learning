\# Lab 5: Frameworks and Automatic Differentiation



\## 1-3



Implemented a scalar automatic differentiation engine.



Implemented:

\- Value node

\- Addition

\- Multiplication

\- Power operation

\- ReLU activation

\- Backpropagation using topological sorting



\---



\## 4. Gradient Verification



Expression:



```python

((a \* b + c) \*\* 2).relu().sum()

```



Results:



Custom Engine:



a grad = -24.0



b grad = 16.0



c grad = 8.0



PyTorch:



a grad = -24.0



b grad = 16.0



c grad = 8.0



The gradients match PyTorch autograd.



\---



\## 5. make\_moons Training



Architecture:



2 → 8 → 1



Activation:

Sigmoid



Loss:

Binary Cross Entropy (BCE)



Median runtime of 5 runs:



0.1922476 seconds



\---



\## 6. Framework Analysis



Tensor properties:



Shape: torch.Size(\[1000, 2])



Dtype: torch.float32



Contiguous: True



Non-contiguous view:



is\_contiguous() = False



Timing results:



Eager execution:

0.1101 s



Traced execution:

0.0997 s



Contiguous tensor:

0.0358 s



Non-contiguous tensor:

0.0161 s



\---



\## Conclusion



The implemented automatic differentiation engine produced the same gradients as PyTorch autograd.



Traced execution was slightly faster than eager execution on this system.



Tensor contiguity can affect operation performance and should be considered when optimizing models.


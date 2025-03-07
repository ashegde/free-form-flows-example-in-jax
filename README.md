# free-form-flows-example-in-jax

In the above code, we take a look at a simple 2D example of the free-form normalizing flows method, which was developed in

Draxler, F., Sorrenson, P., Zimmermann, L., Rousselot, A., & Köthe, U. (2024, April). 
Free-form flows: Make any architecture a normalizing flow. 
In International Conference on Artificial Intelligence and Statistics (pp. 2197-2205). PMLR.

As documented in the reference, the advantage of this formulation is that it avoids the need to explicitly use
invertible neural networks. Rather, "invertibility" is enforced through a reconstruction penalty.

The code here is written in pure JAX and utilizes a simple MLP which is trained using AdamW. 
Despite this, we are able to generate non-trivial results, albeit after substantial hyper-parameter tuning.
For many choices of network settings and hyperparameters, the training process is rather unstable. Some of this
has been resolved by switching to ELU activations. While training can still be unstable, the visual performance is 
now much more consistent.

The results can be regenerated by running,

```
python run_example.py
```

which -- under the current settings -- produces the following result:

![generated_samples](https://github.com/user-attachments/assets/547e3f3a-ff59-4bf6-8c67-9120adaf6e90)


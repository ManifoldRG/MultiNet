## LMMs-Eval for MAGMA

To faciliate the quantitative evaluation of our model, we also provide a model class for [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

After installing lmms-eval, copy 'magma.py' to 'lmms-eval/lmms-eval/models' folder.

Remember to register our model by modifying the 'lmms-eval/lmms_eval/models/__init__.py' file as follows:

```python
AVAILABLE_MODELS = {
    # many previous registered models
    "magma": Magma,
}
```
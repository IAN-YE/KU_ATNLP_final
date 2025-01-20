# KU_ATNLP_final

This is University of Copenhagen Advanced Topics of NLP.

You can use vocab_manager.py to manage the vocabulary of the dataset. The vocab_manager.py is a command line tool that can be used to create, update, and save the vocabulary of the dataset. The vocabulary is saved as a json file.

To run the Experiment Code, you can use the following command to run the code in Code:

```python Experiment_1a.py```

```python Experiment_1b.py ```

And for Experiment 2, Experiment_2.py is used to run the code with the easiest teacher forcing:

```python Experiment_2.py```

For Experiment_2_train.py and Experiment_2_evaluation.py it is used to test deocding with oracele and without oracle length:

```python Experiment_2_evaluation.py```
```python Experiment_2_train.py```

For Experiment_3, Experiment_3.py is used to run the code:

```python Experiment_3.py```

For LLM, you can run ```python finetune_Qwen.py``` to finetune the model, and run ```python Qwen_finetune_infer.py``` to get the output of a finetuned model.

Then you can also run ```python Qwen.py``` to do in-context-learning.
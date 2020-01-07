# Domain Adaptation for Conversation Response Ranking

This repo contains the implementation of two regularization techniques for Deep Matching Networks, built on top of the code from https://github.com/yangliuy/NeuralResponseRanking. We added a parameter to employ either Domain Adversarial Learning (DAL) to induce domain-agnostic representations, or to apply multi-task learning for domain classification (MTL) inducing domain-aware representations. We modified the .config files to receive extra inputs such as the out-of-domain prediction set.

![Image Title](./img/DomainAdversarialLearning_DMN.png)

To enable MTL or DAL use the following parameter with either 'DMN-ADL' or 'DMN-MTL'as input :

```
 python main_conversation_qa.py --domain_training_type '$REGULARIZATION'
```

To see some examples and run the code you can also use this [google colab notebook](https://colab.research.google.com/drive/1BLUFYpNY5_tcyVx0EdEfzpJfoooErZz1) that clones the repo, downloads the dataset and run experiments.

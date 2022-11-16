# Methods for Semi-Supervised Data Augmentation in Sequence Classification Tasks
Our project objective is designing a system
for scalable and accurate unsupervised/semisupervised
label generation for text classification
data. A common data issue across a
variety of contexts and industries is the lack
of well-labeled data and a prominent example
is intent classification, where customers
may send texts/emails about a variety of problems.
With enough labeled intent data, it is
entirely possible to train from scratch or safely
leverage transfer learning. However, obtaining
enough labeled data points for ground-up development
or even fine-tuning can be challenging
in terms of cost and manual effort. Instead,
we aim to investigate generalizable methodologies
for using few-shot and zero-shot for labeling
at scale with limited labeled data.


In few-shot learning literature, Siamese
Neural Network (SNN) (Florian Schroff, 2015) is
popular within the computer vision community and
generally predicts well in small dataset settings.
Pseudo Siamese Neural Network (PSNN) (Xia et al.,
2021) further allows for different weight optimizations
for queries and intents. The core idea is that when we do not have a lot of samples to learn from, we want to exploit the similarity of samples that we do have.
By doing so, we learn clearer decision boundaries in the classification task in the sense of pushing query embeddings of similar intents closer to each other (and away
from other intents).


We adapt a similar structure as PSNN but for text instead of images for few-shot classification. We compare our methodology against several baselines, including zero-shot
learning methodologies and fine-tuning LMs, and found that our PSNN framework performs the best under low-data situations.  

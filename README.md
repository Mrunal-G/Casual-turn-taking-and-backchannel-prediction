# Speech_Conversation_fusion_LLM

- Amazon Alex AI research paper: https://www.amazon.science/publications/turn-taking-and-backchannel-prediction-with-acoustic-and-large-language-model-fusion

1. https://arxiv.org/html/2401.14717v1
2. https://vapi.ai/
3. https://dashboard.vapi.ai/


casual turn-taking and backchannel prediction

- The system should be able to take turns naturally and with minimal latency in a dialogue and
without the need for push-to-talk or wakewords or a period
of silence based on a predefined threshold.

-   Turn-taking benefits remarkably from the fusion, but benefits minimally from the instruction-tuning, while Backchannel shows the opposite trend. (Backchanneling benefits from instruction-tuning and minimally from fusion)

## Datasets
 <!-- https://catalog.ldc.upenn.edu/LDC97S62 -->
 (Switch dataset: release 2)

- Hugggingface dataset: https://huggingface.co/datasets/swda?row=3

**Understand the switchboard dataset**: 
The Switchboard dataset is conisdered a spoken Dialog audio Dataset. The speech data is segmented in utterance units, and text trancriptions provided for each utterance.
`damsl_act_tag`: class label (16, 6) for bh and b which stand for bachchannel examples.
`act_tag`: (list of str) The Dialog Act Tags (separated by ||| in the file). Check Dialog act annotations for more details.

More on tags: 
- https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
- https://compprag.christopherpotts.net/swda.html
- https://convokit.cornell.edu/documentation/switchboard.html#dataset-details

 <!-- Tensorflow: Dataset: https://www.tensorflow.org/datasets/community_catalog/huggingface/swda --> 

[Example of whisper preprocessing switchboard dataset](https://huggingface.co/sanchit-gandhi/whisper-medium-switchboard-5k/blob/main/run_speech_recognition_whisper.py) from huggingface datasets for speech recognition:

```python

swb_disfluencies = ["[noise]", "[laughter]", "[silence]", "[vocalized-noise]", "<a_aside>", "<b_aside>", "<e_aside>",
                        "[laughter-", "_1", "[laugh]", "[sigh]", "[cough]", "[mn]", "[breath]", "[lipsmack]",
                        "[sneeze]", "[skip]", "[pause]", "(%hesitation)", "(%HESITATION)"]
swb_punctuations = ["{", "}", "[", "]-", "]", "((", "))", "(", ")"]

```
- https://isip.piconepress.com/projects/switchboard/
  
# Tasks described in research paper:
1. extendingthe turn-taking model to include backchanneling,
2. use of LLMs with acoustic fusion for these tasks, and
3. exploration of LLMs for instruction-tuning rather than simple token encoding and prediction

## Evaluation Metrics
1. AUC - Area Under Curve
2. Balance Accuracy- bAcc
3. ROC - An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate.
4. EER - Equal Error Rate



# HuBERT fine-tuning for Accoustic classification
The audio files are used for audio classification task with 3 categories with HuBERT. (The recipe described in the given Amazon research paper was done before for my thesis research). They manipulated the HuBERT  architecture for classification by average pooling and using a linear classifier to map the projection to three classes. 
However, I need to get access to the audio files of the dataset and look at the format of the files as the speech data in swithboard is segmented in utterance units. 

# LLM-fine-tuning
LLM funetuning using GPT/ Redpajama models on huggingface used to encode the text of the (partial) utterances.
Then, the embedding is fed into a linear layer of dimension 3
for classification. 
Note: Depending on the base LLM being used, different fine-tuning strategies are applied

The given research paper shows more potential with redpajama. (They haven't specified which variation of redpajama model is used)

  ===
  
  `RedPajama`:
  
- HuggingFace Dataset: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2?row=0
  
HuggingFace Redpajama based chat and instruct models (3B & 7B variations):
- https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
- https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
- https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat
- https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct

<!-- Would need to try different models for chat/instruction-tuning may be using trl.
 https://medium.com/@vi.ai_/fine-tuning-llama-v2-7b-on-google-colab-unleashing-the-full-potential-of-language-modeling-9b9f05c3be35 -->
  
# Late Fusion 
A late fusion mechanism is used where the final embeddings emitted from the AM and LLM are concatenated and fed into a single linear classification layer with dimension 3 for prediction. 
P(Y |XA, XL)

[This description looks like the late fusion recipe usually used where both the probabilites of the classification results from 2 tasks are fused together]

The authors have described 2 late fusion options:
1.  In `Option 1 (Opt1)` both AM and LLM are loaded
from the pretrained library [31] without fine-tuning.
Then, both the fusion layer and the LLM base model undergo domain adaptation
and downstream task training.

2. In `Option 2 (Opt2)`, aside from loading the pretrained AM as in Opt1, the LLM is loaded after standalone fine-tuning as described in Section 2.2.
   Then the LLM branch is also frozen and only the fusion layer is trained.

**The key difference between Opt1 and Opt2 is whether LLM has been fine-tuned
for the downstream task and frozen**
Meaning in opt1 hubert is not finetuned. In opt2, hubert is fine-tuned for classification. I am going to start with option 2 here. Standlone finetuning would most probably give better results than just loading pretrained model.  

# Multi-task instruction fine-tuning
Rather than setting up a three-way classification, each class is handled as a
separate binary classification task. This will later allows us to evaluate performance as three separate detection tasks.

1. task 1: Continuing Speech (0)  --> encoded as (100)
2. task 2: Backchannel (1)        --> encoded as (010)
3. task 3: Turn-taking (2)        --> encoded as (001)

- For each generated sample, if the prepended instruction corresponds to the ground-truth label, i.e. {inst0, sample0}, {inst1, sample1} and {inst2, sample2}, then the corresponding binary label will
be assigned as 1, otherwise 0.
- Each classifier is only in charge of one
corresponding instruction and updates only its parameters, without
being affected by samples augmented by the other two instructions.
BCELoss applied to the classifiers


   ### Instruction-tuning 

During training, each sample will be
augmented three times, with the following respective instructions:
1. Inst 0: “Identify if the current speaker will continue to speak at
the end of the sentence.”;

2. Inst 1: “Identify if another speaker will
backchannel at the end of the sentence.”;

3. Inst 2: “Identify if another speaker will take the turn at the end of the sentence.”

**Note**: All the above steps did not consider `dialogue history` in instruction tuning.

# Dialog History

Additional steps to consider dialog history are given:

two sentences preceding the target partial utterance, with speaker
changes marked, are appended to the task-specific instruction, using
the following format: 
Identify (instruction text): (history with
speaker token). (target sample with speaker token).

(This step needs further investigation). This step is using other information present in the switchboard dataset.







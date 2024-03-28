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
The Switchboard dataset is conisdered a Dialog Audio Dataset. The speech data is segmented in utterance units, and text trancriptions provided for each utterance.
`damsl_act_tag`: class label (16, 6) for bh and b which stand for bachchannel examples.
`act_tag`: (list of str) The Dialog Act Tags (separated by ||| in the file). Check Dialog act annotations for more details.

More on tags: https://web.stanford.edu/~jurafsky/ws97/manual.august1.html

 <!-- Tensorflow: Dataset: https://www.tensorflow.org/datasets/community_catalog/huggingface/swda --> 

  ===
## RedPajama
- HuggingFace Dataset: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2?row=0
HuggingFace Redpajama based chat and instruct models (3B & 7B variations):
- https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
- https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1
- https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat
- https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct
  
  


## Evaluation Metrics
1. AUC - Area Under Curve
2. Balance Accuracy- bAcc
3. ROC - An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate.
4. EER - Equal Error Rate

# Instruction-tuning 

During training, each sample will be
augmented three times, with the following respective instructions:
1) Inst 0: “Identify if the current speaker will continue to speak at
the end of the sentence.”;

3) Inst 1: “Identify if another speaker will
backchannel at the end of the sentence.”;

5) Inst 2: “Identify if another speaker will take the turn at the end of the sentence.”

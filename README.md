# Speech_Conversation_fusion_LLM

1. https://arxiv.org/html/2401.14717v1
2. https://vapi.ai/
3. https://dashboard.vapi.ai/


casual turn-taking and backchannel prediction

- The system should be able to take turns naturally and with minimal latency in a dialogue and
without the need for push-to-talk or wakewords or a period
of silence based on a predefined threshold.

-   Turn-taking benefits remarkably from the fusion, but benefits minimally from the instruction-tuning, while Backchannel shows the opposite trend. (Backchanneling benefits from instruction-tuning and minimally from fusion)

## Dataset
- https://catalog.ldc.upenn.edu/LDC97S62  (Switch dataset: release 2)

- Hugggingface dataset: https://huggingface.co/datasets/swda?row=3
- Tensorflow: Dataset: https://www.tensorflow.org/datasets/community_catalog/huggingface/swda
  


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

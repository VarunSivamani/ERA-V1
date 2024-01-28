# Capstone 

# MultiModal Phi2

<br>

# Task    

You are going to make a multi-modal LLM that can take these inputs:
1. Text
2. Image
3. Audio 

The output remains text for us. Here are the guidelines:
### Training:   

1. Image:
    - Use the original Instruct 150k dataset, and use CLIP to get the image embeddings:   
    - either run it in real-time (need more GPU), or   
    - store the embeddings (basically preprocess and store)      
    - Add your projection layer from this CLIP embeddings to something that can be fed to your Phi Model (do not apply QLoRa to this payer)   
    - Add an adapter that you'll train (QLoRa) on the instruct 150k dataset

<br>

2. Audio:   
    - You need to use Whisper to perform ASR. 
    - Add a projection layer for whisper output (which is text only)
    - This audio part "should" not require any training, but a pipeline from your end to link it properly to your model

3. Text:   
    - You are going to use Microsoft's Phi-2 or any other model and generate data. Recommend you generate this data in parallel, don't generate and store everything as that would be a very very large dataset
    - You are going to collect "some" clean data (100MB when zipped). This data CAN be generated from Phi-2 and stored.
    - You are going to use the same tokenizer and other data structures 
    - You are going to use AWS (or an equvalent system) where you are going to train YOUR model. 
    - You are going to train YOUR model. Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more you have to train!!!
    - Then you're going to take the default Phi-2 model (Microsoft's version and move to the next step)
    - You'll start fine-tuning the model with QLORA (Fine-tuning & Multimodality)


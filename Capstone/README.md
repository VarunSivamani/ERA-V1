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

<br>

3. Text:   
    - You are going to use Microsoft's Phi-2 or any other model and generate data. Recommend you generate this data in parallel, don't generate and store everything as that would be a very very large dataset
    - You are going to collect "some" clean data (100MB when zipped). This data CAN be generated from Phi-2 and stored.
    - You are going to use the same tokenizer and other data structures 
    - You are going to use AWS (or an equvalent system) where you are going to train YOUR model. 
    - You are going to train YOUR model. Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more you have to train!!!
    - Then you're going to take the default Phi-2 model (Microsoft's version and move to the next step)
    - You'll start fine-tuning the model with QLORA (Fine-tuning & Multimodality)

<br>

# Solution

## Areas of Improvement

1. Different loss function can be utilized to better understand the predictions of the model.
2. Performance of the Lighter version of CLIP should be compared with the used version.
3. We have used Instruct 150k dataset for finetuning the model. The original Llava model was finetuned on Blip Captions. Some more datasets can also be used for better finetuning performance.
4. Different Audio libraries can also be compared and tested on speech recognition.

<br>

# Pre-training Logs

```python
iter 0 step 0: loss 12.5670, LR: 0.000000, iter time: 1510.67ms
iter 100 step 25: loss 7.9562, LR: 0.000300, iter time: 107.06ms
iter 200 step 50: loss 7.7976, LR: 0.000600, iter time: 95.83ms
iter 300 step 75: loss 8.1913, LR: 0.000900, iter time: 95.73ms
iter 400 step 100: loss 9.1314, LR: 0.001200, iter time: 97.15ms
iter 500 step 125: loss 8.7444, LR: 0.001500, iter time: 97.29ms
iter 600 step 150: loss 9.1133, LR: 0.001800, iter time: 97.54ms
iter 700 step 175: loss 7.8593, LR: 0.002100, iter time: 96.22ms
iter 800 step 200: loss 8.6967, LR: 0.002400, iter time: 98.89ms
iter 900 step 225: loss 7.7239, LR: 0.002700, iter time: 95.66ms
iter 1000 step 250: loss 8.4533, LR: 0.003000, iter time: 97.23ms
iter 1100 step 275: loss 6.2117, LR: 0.003300, iter time: 96.62ms
iter 1200 step 300: loss 9.0013, LR: 0.003600, iter time: 93.58ms
iter 1300 step 325: loss 7.8748, LR: 0.003900, iter time: 96.73ms
iter 1400 step 350: loss 8.5645, LR: 0.004200, iter time: 95.93ms
iter 1500 step 375: loss 8.5179, LR: 0.004500, iter time: 96.44ms
iter 1600 step 400: loss 8.0121, LR: 0.004800, iter time: 94.96ms
iter 1700 step 425: loss 8.5510, LR: 0.005100, iter time: 94.43ms
iter 1800 step 450: loss 7.7642, LR: 0.005400, iter time: 96.84ms
iter 1900 step 475: loss 7.2426, LR: 0.005700, iter time: 95.84ms
iter 2000 step 500: loss 7.6280, LR: 0.006000, iter time: 91.55ms
iter 2100 step 525: loss 8.4805, LR: 0.005998, iter time: 95.05ms
iter 2200 step 550: loss 7.3383, LR: 0.005991, iter time: 96.98ms
iter 2300 step 575: loss 7.6553, LR: 0.005979, iter time: 97.88ms
iter 2400 step 600: loss 7.6904, LR: 0.005963, iter time: 96.96ms
iter 2500 step 625: loss 8.0117, LR: 0.005942, iter time: 95.29ms
iter 2600 step 650: loss 8.2796, LR: 0.005917, iter time: 96.71ms
iter 2700 step 675: loss 6.5858, LR: 0.005887, iter time: 96.51ms
iter 2800 step 700: loss 7.7028, LR: 0.005853, iter time: 96.25ms
iter 2900 step 725: loss 7.5999, LR: 0.005815, iter time: 97.89ms
iter 3000 step 750: loss 7.2146, LR: 0.005772, iter time: 97.59ms
```

<br>

# Projection Layer Training Logs

```python
Epoch : 1/15
Loss  : 7.348170757293701
Caption    :  ['A tray holding a sandwich and cappuccino, next to the pastry.']
Prediction :  [',-.s, of the few and appuccino. with to a coffee case\nThe\nTheTheTheThe\n\nThe\n\nThe\n\nTheThe\n\n\n\n\n\n\nTheThe\n\n\n\n\n\n\nThe']
==============================
Epoch : 2/15
Loss  : 7.897108554840088
Caption    :  ['two zebras are standing together in the woods']
Prediction :  [',isea,,bras, two in, a middle.The']
==============================
Epoch : 3/15
Loss  : 6.406453609466553
Caption    :  ['A woman in a hat sitting next to luggage.']
Prediction :  [",,es,,'s the red and on to a,\nThe"]
==============================
Epoch : 4/15
Loss  : 6.641895294189453
Caption    :  ['Two Zebras grazing together in a grassy area.']
Prediction :  [',-es,,ekan, on, a fieldland field.\nThe']
==============================
Epoch : 5/15
Loss  : 6.686727523803711
Caption    :  ['A man prepares to cross the street at a crosswalk']
Prediction :  [',ing.,, with to be a street. a crosswalk.TheThe']
==============================
Epoch : 6/15
Loss  : 6.352148532867432
Caption    :  ['Two Zebras grazing together in a grassy area.']
Prediction :  [',ardsers:Dresas, in, a fieldy field.\nThe']
==============================
Epoch : 7/15
Loss  : 6.285255432128906
Caption    :  ['A toilet sitting underneath a medicine cabinet in a bathroom.']
Prediction :  [',idede(, is on a bed cabinet, a bathroom.']
==============================
Epoch : 8/15
Loss  : 6.199134349822998
Caption    :  ['A picture of a bunch of bananas sitting on a table. ']
Prediction :  ['.ose,:[ of the man of plants with on a table with']
==============================
Epoch : 9/15
Loss  : 5.938331127166748
Caption    :  ['A couple of computer monitors sitting on top of a wooden desk.']
Prediction :  [',olss, of years programs, on a of each desk desk.']
==============================
Epoch : 10/15
Loss  : 5.628418445587158
Caption    :  ['A clock that is hanging underneath a glass arch.']
Prediction :  [' thehersss:wise is not on the table paneway']
==============================
Epoch : 11/15
Loss  : 5.419788360595703
Caption    :  ['A woman in black jacket watching a cat eating from pizza box.']
Prediction :  [',lingts, is the a and a man on a a box.\nA']
==============================
Epoch : 12/15
Loss  : 5.401681900024414
Caption    :  ['A bedroom with a bed and small table near by.']
Prediction :  [' theol\n::, a window, a table, the.']
==============================
Epoch : 13/15
Loss  : 4.137636184692383
Caption    :  ['A couple of computer monitors sitting on top of a wooden desk.']
Prediction :  [',ardsless, of weeks systems, on a of each desk table,\nA']
==============================
Epoch : 14/15
Loss  : 3.991928815841675
Caption    :  ['A toilet sitting underneath a medicine cabinet in a bathroom.']
Prediction :  [',hedts,\n on the sink cabinet\n a bathroom']
==============================
Epoch : 15/15
Loss  : 3.35038685798645
Caption    :  ['Boats docked on land sitting side by side next to a lake.']
Prediction :  [',ards\n.ond[urations at the\n on by side\n to each boat']
==============================
```

<br>

# QLoRA Fine-tuning Logs

```python
[500/500 07:37, Epoch 0/1]
Step	Training Loss
10	1.373200
20	1.475100
30	1.800600
40	1.864200
50	2.059700
60	1.420600
70	1.390700
80	1.728400
90	2.142600
100	2.375900
110	1.837600
120	1.260800
130	1.418500
140	1.899400
150	1.761400
160	1.119000
170	1.384200
180	1.521800
190	2.065300
200	1.840700
210	1.450700
220	1.542600
230	1.239800
240	1.841600
250	2.105200
260	1.164700
270	1.128000
280	1.386900
290	1.879100
300	2.163400
310	1.733300
320	1.551800
330	1.499800
340	2.255200
350	2.332400
360	1.190300
370	1.715900
380	1.948700
390	1.967400
400	1.967500
410	1.382100
420	1.493700
430	1.934500
440	2.124700
450	2.074700
460	1.232700
470	1.310700
480	1.264300
490	1.817800
500	2.142300

TrainOutput(global_step=500, training_loss=1.6916268558502197, metrics={'train_runtime': 501.7476, 'train_samples_per_second': 0.997, 'train_steps_per_second': 0.997, 'total_flos': 1971890728980480.0, 'train_loss': 1.6916268558502197, 'epoch': 0.06})
```
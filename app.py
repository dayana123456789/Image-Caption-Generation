from flask import Flask,render_template,request
import cv2
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing import image,sequence
from tqdm import tqdm
import torch
from transformers import VisionEncoderDecoderModel,AutoTokenizer,ViTFeatureExtractor
import cv2
from PIL import Image
import torch

vocab=np.load("vocab.np",allow_pickle=True)
vocab=vocab.item()
inv_vocab={v:k for k,v in vocab.items()}
vocab_size=len(vocab)
max_length=40

model_name="bipin/image-caption-generator"
model=VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor=ViTFeatureExtractor.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained("gpt2")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pixel_values=feature_extractor(images=[image],return_tensors="pt").pixel_values
pixel_values=pixel_values.to(device)
max_length=128
#get model predictions
output=model.generate(pixel_values,num_beams=4,max_length=max_length)
preds=tokenizer.decode(output[0],skip_special_tokens=True)
print(preds)
cv2.imshow(image)
app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"]=1
@app.route("/")
def  index():
    return render_template("index.html")
@app.route("/after",methods=["GET","POST"])
def after():
    global model,vocab,inv_vocab
    img=request.files["file1"]
    img.save("static/file.jpg")
    print("="*"5")
    print("Image saved")


    image=cv2.imread("static/file.jpg")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(224,224))
    image=np.reshape(image,(1,224,224,3))

    incept=model.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")

    text_in=['startofseq']

    final=""
    print("="*50)
    print("Getting captions")

    count=0
    while tqdm(count<20):
        count+=1
        encoded=[]
        for i in text_in:
            encoded.append(vocab[i])
        padded=pad_sequences([encoded],maxlen=max_length,padding="post",truncating="post")
        sampled_index=np.argmax(model.predict({incept,padded}))
        sampled_word=inv_vocab[sampled_index]

        if sampled_word!='endofseq':
            final=final+""+sampled_word
        text_in.append(sampled_word)
    return render_template("after.html",data=final)  
if __name__=="__main__":
   app.run(debug=True)
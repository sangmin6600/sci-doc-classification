import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from peft import PeftModel

model_name = 'beomi/llama-2-ko-7b'
patent_peft_model_path = '/home/smkim/PatentClassification/llama-2-ko-7b-ft-patent_classification/checkpoint-2812'
paper_peft_model_path = '/home/smkim/PatentClassification/llama-2-ko-7b-ft-paper_classification_ct/checkpoint-1406'

tokenizer = AutoTokenizer.from_pretrained(model_name)
patent_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
patent_model = PeftModel.from_pretrained(patent_model, patent_peft_model_path).to('cuda')
patent_model=patent_model.merge_and_unload()
paper_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
paper_model = PeftModel.from_pretrained(paper_model, paper_peft_model_path).to('cuda')
paper_model=paper_model.merge_and_unload()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# def predict(message, history):
#     history_transformer_format = history + [[message, ""]]
#     stop = StopOnTokens()

#     messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
#                 for item in history_transformer_format])

#     model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
#     streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
#     generate_kwargs = dict(
#         model_inputs,
#         streamer=streamer,
#         max_new_tokens=256,
#         do_sample=True,
#         top_p=0.92,
#         top_k=40,
#         # temperature=1.0,
#         no_repeat_ngram_size=5,
#         num_beams=5,
#         stopping_criteria=StoppingCriteriaList([stop])
#         )
#     t = Thread(target=model.generate, kwargs=generate_kwargs)
#     t.start()

#     partial_message = ""
#     for new_token in streamer:
#         if new_token != '<':
#             partial_message += new_token
#             yield partial_message

def predict(message, history):
    # 사용자 입력 분석 및 모델 선택
    if "CPC" in message or "cpc" in message:
        model = patent_model
        instruction = "Analyze this document and provide the corresponding technical field code according to the CPC classification."
    elif "citation topic" in message or "Citation topic" in message or "Citation Topic" in message:
        model = paper_model
        # instruction = "Analyze the content of this document and identify the relevant academic field according to the WoS category."
        instruction = "Read through the document and specify the Citation Topics that are most relevant to its findings and discussions."
    else:
        model = patent_model  # 기본값 설정
        instruction = "Analyze this document and provide the corresponding technical field code according to the CPC classification."

    # history_transformer_format = history + [[message, ""]]
    # messages = "".join(["\n<human>:" + item[0] + "\n<bot>:" + item[1] for item in history_transformer_format])
    
    # 입력 텍스트 형식 구성
    input_text = f"""[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{message}\n\n{instruction} [/INST]\n\n"""

    model_inputs = tokenizer(input_text, return_tensors="pt", max_length=4096, truncation=True).to("cuda")

    output = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.92,
        top_k=40,
        no_repeat_ngram_size=5,
        num_beams=5,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    new_message = generated_text[len(input_text):]  # 이전 메시지를 제외한 새로운 부분만 추출
    return new_message

gr.ChatInterface(predict).launch(share=True)
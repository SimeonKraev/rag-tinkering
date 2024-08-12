import os
import torch
from dotenv import load_dotenv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
READER_MODEL_NAME = os.getenv('READER_MODEL_NAME')
HUG_TOKEN = os.getenv('HUG_TOKEN')


def reader_model(context, prompt):
    prompt_in_chat_format = [{"role": "system",
                              "content": "Using the information contained in the context and its documents, give a comprehensive answer to the question. Respond only to the question asked, response should be clear, unrepetative and relevant to the question."},
                             {"role": "user",
                              "content": f"Context:{context} - here is the question you need to answer. Question: {prompt}"}]

    # for quantization, not used atm
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME,
                                                 device_map="auto",
                                                 torch_dtype="auto",
                                                 trust_remote_code=True,
                                                 token=HUG_TOKEN,
                                                 offload_folder="offload")  #, quantization_config=bnb_config)

    #model.to('cpu')
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, padding_side='left')

    READER_LLM = pipeline(model=model,
                          tokenizer=tokenizer,
                          task="text-generation",
                          do_sample=True,
                          #temperature=0.7,
                          repetition_penalty=1.1,
                          return_full_text=False,
                          min_new_tokens=50,  #)
                          max_new_tokens=400)

    conversation_history = "\n".join([f"{message['role']}: {message['content']}" for message in prompt_in_chat_format])
    tokenizer.chat_template = conversation_history

    answer = READER_LLM(conversation_history)[0]["generated_text"]

    return answer

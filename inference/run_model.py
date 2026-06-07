import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

CUDA = torch.cuda.is_available()


def _load_kwargs():
    if CUDA:
        return dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
        )
    else:
        # CPU: load in float16, no bitsandbytes needed (~6GB RAM)
        return dict(
            dtype=torch.float16,
            device_map="cpu",
        )


def load_base_model(model_name: str = "ministral/Ministral-3b-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **_load_kwargs(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned_model(
    base_model_name: str = "ministral/Ministral-3b-instruct",
    adapter_path: str = "./outputs/model",
):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        **_load_kwargs(),
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()

import json
import torch
from transformers import BlenderbotTokenizer, BlenderbotSmallTokenizer
from libs.utils import norm
from module.model import MyModel
from libs.config import Config
from libs.utils.model_loader import get_tokenizer


PRETRAIN_MODEL = "/home/trandat/Documents/KLTN/epoch_0.ckpt"


tokenizer = get_tokenizer()


model = MyModel.from_pretrained(PRETRAIN_MODEL)
model.tie_tokenizer(tokenizer)
model.eval()


with open("/home/trandat/Documents/KLTN/test.json", "r") as f:
    data = json.load(f)

    process = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
    dialog = data.get("dialog", [])
    if not dialog:
        raise ValueError("Dialog data is empty or missing.")

    inputs, context, knowledge = [], [], []

    for i, turn in enumerate(dialog):
        text = process(norm(turn["text"]))

        if turn["speaker"] == "sys":
            strat_id = process("[" + turn["strategy"] + "]")

            heal = process(turn["heal"])
        else:
            knowledge = process(turn["knowledge"])

        if i > 0 and turn["speaker"] == "sys":
            inputs.append(
                {
                    "context": context.copy(),
                    "knowledge": knowledge + heal,
                    # "knowledge": [],
                    "response": text,
                    "strat_id": strat_id,
                }
            )
        context = context + [text]

    inputs = inputs[-1]

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

    context = [c + [eos] for c in context]
    context += [knowledge + [eos]]
    input_ids = sum(context, [])[-(Config.MAX_INPUT_LENGTH) :]

    labels = (strat_id + inputs["response"] + [eos])[
        : Config.MAX_DECODER_INPUT_LENGTH + 1
    ]
    decoder_input_ids = [bos] + labels[:-1]

decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
attention_mask = torch.tensor([1.0] * len(input_ids), dtype=torch.float)


with torch.no_grad():
    _, output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )


generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print("Generated Text:", generated_text[0])

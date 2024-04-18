from model import ModelArgs, Transformer
import torch

from tokenizer import Tokenizer

from config import args

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = r"model_checkpoints\model.pth"

model = Transformer(args)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

idx2bytes_path = r"tokenizer_checkpoints\idx2bytes.pkl"
pair2idx_path = r"tokenizer_checkpoints\pair2idx.pkl"

tokenizer = Tokenizer.from_pickle_files(idx2bytes_path=idx2bytes_path, pair2idx_path=pair2idx_path)

def auto_regressive_generation(text, max_tokens):
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens_tensor[:, -128:], penalty=1.1)
            output = logits.argmax(dim=-1)[0, -1].unsqueeze(-1).unsqueeze(-1)
            tokens_tensor = torch.cat((tokens_tensor, output), dim=-1)
            if output.item() == 0:
                break
    output_text = tokenizer.decode(tokens_tensor[0].tolist())
    return output_text


if __name__ == "__main__":
    while(True):
        procceed = int(input("Proceed(1/0)?\n"))
        if procceed == 0:
            break

        num_tokens = int(input("Enter tokens: "))
        text = input("Enter text prompt:\n")
        print()

        output = auto_regressive_generation(text, num_tokens)
        for character in output:
            print(character, end='')
            time.sleep(0.01)
        print()
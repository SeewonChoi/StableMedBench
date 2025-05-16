from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
import torch.nn as nn


class CustomGPT(nn.Module):
    def __init__(self, vocab_size=1969, hidden_size=768, full=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.resize_token_embeddings(vocab_size)

        if not full:
            for param in self.model.parameters(): 
                param.requires_grad = False

            for i in range(-2, 0):  # Last two blocks
                for param in self.model.base_model.h[i].parameters():
                    param.requires_grad = True
        
        self.time_layer = nn.Linear(1, hidden_size)
        self.combine_layer = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, input_ids, times, attention_mask=None):
        assert(input_ids.max().item() < self.vocab_size)  
        assert(input_ids.min() >= 0)

        assert(self.model.base_model.wte.num_embeddings == self.vocab_size)
        token_embeddings = self.model.base_model.wte(input_ids)
        time_embeddings = self.time_layer(times.unsqueeze(-1))
        embs = token_embeddings + time_embeddings
        
        output = self.model(
            inputs_embeds = embs,
            attention_mask=attention_mask,
            return_dict=True)

        logits = output.logits # [batch_size, seq_len, vocab_size]
        return logits

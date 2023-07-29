import torch
import transformers


# sample_tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# sample_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sample_2d[:, 0, :].squeeze())
# print(sample_2d[:, 0, :])

device = torch.device("mps")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.AutoModel.from_pretrained("bert-base-uncased").to(device)

toks = tokenizer(["Hello world!", "blah"], return_tensors="pt", padding="longest").to(device)
model_out = model(**toks)

print(model_out.last_hidden_state)
print(model_out.last_hidden_state[:, 0, :].squeeze())

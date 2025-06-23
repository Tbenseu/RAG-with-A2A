from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"
save_directory = "./"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download and save the model
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_directory)

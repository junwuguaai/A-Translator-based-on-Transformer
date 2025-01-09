from datasets import Dataset, load_dataset, load_from_disk
dataset = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
dataset.save_to_disk("dataset/opus_books") # 保存到该目录下
dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = '[SOS]'
        if self.sos_token not in tokenizer_src.get_vocab():
            tokenizer_src.add_tokens([self.sos_token])

        # Get the token ID for [SOS]
        sos_token_id = tokenizer_src.token_to_id(self.sos_token)
        self.sos_token = torch.tensor(sos_token_id, dtype=torch.int64)

        self.eos_token = '[EOS]'
        if self.eos_token not in tokenizer_src.get_vocab():
            tokenizer_src.add_tokens([self.eos_token])

        # Get the token ID for [EOS]
        eos_token_id = tokenizer_src.token_to_id(self.eos_token)
        self.eos_token = torch.tensor(eos_token_id, dtype=torch.int64)

        self.pad_token = '[PAD]'
        if self.pad_token not in tokenizer_src.get_vocab():
            tokenizer_src.add_tokens([self.pad_token])

        # Get the token ID for [PAD]
        pad_token_id = tokenizer_src.token_to_id(self.pad_token)
        self.pad_token = torch.tensor(pad_token_id, dtype=torch.int64)


        self.unk_token = "[UNK]"
        if self.unk_token not in tokenizer_src.get_vocab():
            tokenizer_src.add_tokens([self.unk_token])

        # Get the token ID for [UNK]
        unk_token_id = tokenizer_src.token_to_id(self.unk_token)
        self.unk_token = torch.tensor(unk_token_id, dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index:any) -> any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')


        #Add SOS and EOS to the source text
        encoder_input = torch.cat(
            (
                self.sos_token.unsqueeze(0),
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
            )
        )

        #Add SOS to the decoder input
        decoder_input = torch.cat(
            (
                self.sos_token.unsqueeze(0),
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            )
        )

        #Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            (
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token.unsqueeze(0),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            )
        )

        # print(f'encoder_input_size:', encoder_input.size(0))
        # print(f'decoder_input_size:', decoder_input.size(0))
        # print(f'label size:', label.size(0))
        # print(f'seq_len:',self.seq_len)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            "encoder_input":encoder_input,#(Seq_Len)
            "decoder_input":decoder_input,#(Seq_Len)
            "encoder_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),#(1,1,Seq_Len)
            "decoder_mask":(decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)&causal_mask(decoder_input.size(0)),#(1,Seq_Len) & (1,Seq_Len,Seq_Len)
            "label":label,#(Seq_Len)
            "src_text":src_text,
            "tgt_text":tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask == 0
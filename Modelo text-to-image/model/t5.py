import torch
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel

MAX_LENGTH = 512

# Variants: https://huggingface.co/docs/transformers/model_doc/t5v1.1. 1.1 versions must be finetuned.
T5_version = {'tokenizer': None, 'model': None, 'handle': 't5-small', 'dim': 512, 'size': .24}

# Fast tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
def _check_downloads():
    if T5_version['tokenizer'] is None:
        T5_version['tokenizer'] = T5Tokenizer.from_pretrained(T5_version['handle'])
    if T5_version['model'] is None:
        T5_version['model'] = T5EncoderModel.from_pretrained(T5_version['handle'])


def t5_encode_text(text, max_length=MAX_LENGTH):

    _check_downloads()
    tokenizer = T5_version['tokenizer']
    model = T5_version['model']
    # Move to cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # Tokenize text
    tokenized = tokenizer.batch_encode_plus(
        text,
        padding='longest',
        max_length=max_length,
        truncation=True,
        return_tensors="pt",  # Returns torch.tensor instead of python integers
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()

    # Don't need gradient - T5 frozen during Imagen training
    with torch.no_grad():
        t5_output = model(input_ids=input_ids, attention_mask=attention_mask)
        final_encoding = t5_output.last_hidden_state.detach()

    # Wherever the encoding is masked, make equal to zero
    final_encoding = final_encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)

    return final_encoding, attention_mask.bool()


def get_encoded_dim() -> int:
    """
    Gets the encoding dimensionality of a given T5 encoder.
    """
    return T5_version['dim']

def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    """
    For classifier free guidance. Creates a boolean mask for given input shape and probability of `True`.

    :param shape: Shape of mask.
    :param prob: Probability of True. In interval [0., 1.].
    :param device: Device to put the mask on. Should be the same as that of the tensor which it will be used on.
    :return: The mask.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
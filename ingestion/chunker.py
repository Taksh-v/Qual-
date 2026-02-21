import uuid
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 75
):
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # overlap
            overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

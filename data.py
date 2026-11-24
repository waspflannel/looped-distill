import copy
import gc
import hashlib
import queue
import threading

import torch
from datasets import load_dataset

from config import FINEWEB, FINEWEB_REVISION

DOCUMENT_END = "<|endoftext|>"


def is_held_out(document, every):
    return int(hashlib.sha256(document["id"].encode()).hexdigest()[:8], 16) % every == 0


class PackedStream:
    def __init__(self, tokenizer, config, held_out, state=None):
        self.tokenizer = tokenizer
        self.document_end = tokenizer.convert_tokens_to_ids(DOCUMENT_END)
        self.held_out = held_out
        self.held_out_every = config["held_out_every"]
        self.documents = load_dataset(
            FINEWEB, name=config["fineweb_config"], split="train", streaming=True, revision=FINEWEB_REVISION
        )
        self.leftover = []
        if state is not None:
            self.documents.load_state_dict(state["documents"])
            self.leftover = list(state["leftover"])
        self.reader = iter(self.documents)

    def next_batch(self, rows, length):
        needed = rows * length
        while len(self.leftover) < needed:
            document = next(self.reader)
            if is_held_out(document, self.held_out_every) == self.held_out:
                self.leftover += self.tokenizer(document["text"])["input_ids"] + [self.document_end]
        batch, self.leftover = self.leftover[:needed], self.leftover[needed:]
        return torch.tensor(batch).view(rows, length)

    def state(self):
        return {"documents": copy.deepcopy(self.documents.state_dict()), "leftover": list(self.leftover)}

    def close(self):
        if self.reader is not None and hasattr(self.reader, "close"):
            self.reader.close()
        self.reader = None
        self.documents = None
        gc.collect()


class PrefetchStream:
    def __init__(self, stream, rows, length, pin_memory, prefetch=4):
        self.stream = stream
        self.rows = rows
        self.length = length
        self.pin_memory = pin_memory
        self._last_state = stream.state()
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=prefetch)
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _produce(self):
        try:
            while not self._stop.is_set():
                batch = self.stream.next_batch(self.rows, self.length)
                if self.pin_memory:
                    batch = batch.pin_memory()
                self._put(("batch", batch, self.stream.state()))
        except Exception as error:
            self._put(("error", error, None))

    def next_batch(self):
        kind, payload, snapshot = self._queue.get()
        if kind == "error":
            raise payload
        self._last_state = snapshot
        return payload

    def state(self):
        return self._last_state

    def close(self):
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)
        self.stream.close()

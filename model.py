import json
import os
from pathlib import Path
from uuid import uuid4

from chonkie import RecursiveChunker
from dotenv import load_dotenv
from google import genai


class GeminiEmbedding2:
    def __init__(self):
        self.id = 'gemini-embedding-2'
        self.max_tokens = 8192
        # gemini-embedding-2 ignores the API's task_type parameter, so the task
        # is given as a prefix on the text instead.
        self.task = 'SEMANTIC_SIMILARITY'
        self.prefix = 'task: sentence similarity | query: '
        load_dotenv(Path(os.environ['BUILD_WORKING_DIRECTORY']) / '.env')
        self.client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def embed(self, text: str) -> list[str]:
        # Each chunk gets its own UUID so that IDs never collide across
        # sources. Callers persist the UUIDs alongside their own data.
        return [self._embed_chunk(chunk) for chunk in self._chunk(text)]

    def delete(self, uuids: list[str]) -> None:
        for uuid in uuids:
            self._path(uuid).unlink()

    def _chunk(self, text: str) -> list[str]:
        # The API silently ignores input past the token limit, so long text is
        # split up front. Sizes are measured in UTF-8 bytes because every token
        # covers at least one byte, so a chunk that fits in bytes fits in tokens.
        size = self.max_tokens - len(self.prefix.encode())
        if len(text.encode()) <= size:
            return [text]
        chunker = RecursiveChunker(tokenizer='byte', chunk_size=size)
        return [chunk.text for chunk in chunker.chunk(text)]

    def _embed_chunk(self, text: str) -> str:
        uuid = str(uuid4())
        print(f'Embedding: {uuid}', flush=True)
        data = {self.id: {self.task: self._encode(text)}}
        path = self._path(uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return uuid

    def _path(self, uuid: str) -> Path:
        return (
            Path(os.environ['BUILD_WORKING_DIRECTORY'])
            / 'data'
            / 'embedding'
            / f'{uuid}.json'
        )

    def _encode(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.id,
            contents=f'{self.prefix}{text}',
        )
        return result.embeddings[0].values

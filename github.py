import json
import os
from pathlib import Path
import shutil
import subprocess

from model import GeminiEmbedding2


class GitHub:
    def __init__(self):
        self.cli = shutil.which('gh')

    def get(self, endpoint: str) -> list[dict] | dict:
        # TODO: Fallback to the GitHub REST API if the gh CLI is not available.
        cmd = [
            self.cli,
            'api',
            endpoint,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)


class Comment:
    def __init__(
        self,
        id: int,
        parent: int | None,
        body: str,
        timestamp: str,
        model: GeminiEmbedding2,
        previous: dict | None = None,
    ):
        self.id = id
        self.parent = parent
        self.body = body
        self.timestamp = timestamp
        self.model = model
        self.uuids = self._embed(previous)

    def _embed(self, previous: dict | None) -> list[str]:
        # Reuse the saved embeddings when the text is unchanged, so refetching
        # a discussion only re-embeds the comments that were actually edited.
        if previous is not None and previous['body'] == self.body:
            return previous['uuids']
        if previous is not None:
            self.model.delete(previous['uuids'])
        return self.model.embed(self.body)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'parent': self.parent,
            'body': self.body,
            'timestamp': self.timestamp,
            'uuids': self.uuids,
        }


class Discussion:
    def __init__(
        self,
        owner: str,
        name: str,
        number: int,
        title: str,
        body: str,
        timestamp: str,
        url: str,
        client: GitHub,
        model: GeminiEmbedding2,
    ):
        self.owner = owner
        self.name = name
        self.number = number
        self.title = title
        self.body = body
        self.timestamp = timestamp
        self.url = url
        self.client = client
        self.model = model
        self.path = (
            Path(os.environ['BUILD_WORKING_DIRECTORY'])
            / 'data'
            / 'github'
            / self.owner
            / self.name
            / 'discussion'
            / f'{self.number}.json'
        )
        self.previous = self._load()
        if self._is_stale():
            print(f'Fetching: {self.url}', flush=True)
            self.comments = self._fetch_comments()
            self.uuids = self._embed()
            self._prune()
            self._save()
        else:
            self.comments = self._load_comments()
            self.uuids = self.previous['uuids']

    def _load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def _is_stale(self) -> bool:
        # GitHub bumps the discussion-level timestamp on any activity in the
        # discussion (body edits, new or edited comments, accepted answers),
        # so an unchanged timestamp means nothing in the discussion changed.
        if self.previous is None:
            return True
        return self.previous['timestamp'] != self.timestamp

    def _load_comments(self) -> list[Comment]:
        return [
            Comment(
                id=item['id'],
                parent=item['parent'],
                body=item['body'],
                timestamp=item['timestamp'],
                model=self.model,
                previous=item,
            )
            for item in self.previous['comments']
        ]

    def _fetch_page(self, page: int) -> list[dict]:
        endpoint = (
            f'repos/{self.owner}/{self.name}/discussions/{self.number}/comments'
            f'?per_page=100&page={page}'
        )
        return self.client.get(endpoint)

    def _fetch_comments(self) -> list[Comment]:
        previous = {}
        if self.previous is not None:
            previous = {item['id']: item for item in self.previous['comments']}
        comments: list[Comment] = []
        page = 1
        while True:
            items = self._fetch_page(page)
            if not items:
                break
            for item in items:
                comments.append(
                    Comment(
                        id=item['id'],
                        parent=item['parent_id'],
                        body=item['body'],
                        timestamp=item['updated_at'],
                        model=self.model,
                        previous=previous.get(item['id']),
                    )
                )
            page += 1
        return comments

    def _embed(self) -> list[str]:
        # The discussion timestamp changes on any activity, so compare the
        # text itself to decide whether the body needs re-embedding.
        if (
            self.previous is not None
            and self.previous['title'] == self.title
            and self.previous['body'] == self.body
        ):
            return self.previous['uuids']
        if self.previous is not None:
            self.model.delete(self.previous['uuids'])
        return self.model.embed(f'{self.title}\n\n{self.body}')

    def _prune(self) -> None:
        # Delete the embeddings of comments that no longer exist.
        if self.previous is None:
            return
        active = {comment.id for comment in self.comments}
        for item in self.previous['comments']:
            if item['id'] not in active:
                self.model.delete(item['uuids'])

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'body': self.body,
            'timestamp': self.timestamp,
            'url': self.url,
            'uuids': self.uuids,
            'comments': [comment.to_dict() for comment in self.comments],
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2))


class Repo:
    def __init__(
        self,
        owner: str,
        name: str,
        model: GeminiEmbedding2,
        start: str | None = None,
        discussions: bool = True,
    ):
        self.owner = owner
        self.name = name
        self.model = model
        self.start = start
        self.client = GitHub()
        self.discussions = self._fetch_discussions() if discussions else []

    def _fetch_page(self, page: int) -> list[dict]:
        endpoint = (
            f'repos/{self.owner}/{self.name}/discussions'
            f'?per_page=100&page={page}'
        )
        return self.client.get(endpoint)

    def _build_discussion(self, item: dict) -> Discussion:
        return Discussion(
            owner=self.owner,
            name=self.name,
            number=item['number'],
            title=item['title'],
            body=item['body'],
            timestamp=item['updated_at'],
            url=item['html_url'],
            client=self.client,
            model=self.model,
        )

    def _fetch_discussions(self) -> list[Discussion]:
        # TODO: Remove this hardcoded discussion once early development is done.
        endpoint = f'repos/{self.owner}/{self.name}/discussions/14652'
        return [self._build_discussion(self.client.get(endpoint))]
        discussions: list[Discussion] = []
        page = 1
        while True:
            items = self._fetch_page(page)
            if not items:
                break
            for item in items:
                if self.start and item['updated_at'] < self.start:
                    continue
                discussions.append(self._build_discussion(item))
                return discussions
            page += 1
        return discussions

import json
import os
from pathlib import Path

import github
from model import GeminiEmbedding2


def _load_project() -> dict:
    root = Path(os.environ['BUILD_WORKING_DIRECTORY'])
    return json.loads((root / 'project.json').read_text())


def main():
    project = _load_project()
    model = GeminiEmbedding2()
    for entry in project.get('github', []):
        repo = github.Repo(
            owner=entry['owner'],
            name=entry['name'],
            model=model,
            start=entry.get('start'),
            discussions=entry.get('discussions', True),
        )
        print(f'Discussions: {len(repo.discussions)}')
        for discussion in repo.discussions:
            print(f'  #{discussion.number}: {discussion.title}')
            print(f'  Comments: {len(discussion.comments)}')


if __name__ == '__main__':
    main()

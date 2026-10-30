* goal: hybrid search across various common data sources
* always invoke bazelisk via the local `./bazelisk` script
* update deps: `./bazelisk run :deps.update`
* do everything through bazelisk e.g. `./bazelisk run :github`
* never invoke `python` or `python3` directly
* default to single quotes for strings not double quotes
* never produce overly defensive code that leads to silent failures
* domain data should usually be given its own class e.g. a repo in github
  gets a Repo class and we initialize as much as possible in `__init__`
* decompose long functions into smaller helper functions/methods so each has a
  single responsibility

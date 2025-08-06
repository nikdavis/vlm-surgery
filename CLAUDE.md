This is an example of how we like to build Python here.

# Python project overview
We use uv. Dependencies found in pyproject.toml

Can run modules generally via `uv run python -m src.folder.file`

uv sync as needed

# Coding

We generally write async code as long as our libraries support it.

We typically use httpx for rest calls / clients

We use pydantic models for most data transfer between classes, rest clients etc.

put rest clients in ./src/clients
put data models in ./src/models
general files in ./src or more deeply nested as the ideas surface (following rule of 3 for e.g.)

# testing
we use pytest to test

tests go in ./tests

we organize tests using same path / file name (with test_<file>.py).

e.g. ./src/models/user.py -> ./tests/models/test_user.py

src

├── __init__.py

├── api

│   ├── __init__.py

│   └── scrape_api.py

├── clients

│   └── ocr_client.py

├── models

│   └── user.py

├── src

│   └── stuff.py

├── tests

│   └── test_stuff.py
.. etc

additional notes:
- we don't do any funny business in __init__.py files (we leave them blank! just for module discovery)


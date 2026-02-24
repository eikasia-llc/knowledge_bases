---
description: Look up and read knowledge base documents with automatic dependency resolution
---

// turbo-all

1. Read `dependency_registry.json` from the project root. Each entry has `path`, `type`, `description`, and `dependencies`.

2. Scan the `description` fields to identify entries relevant to the user's question. If the user specified a file, find it by `path`.

3. For the target entry, read its `dependencies` map. Resolve depth-first: if file A depends on B and B depends on C, read C → B → A.

4. Use `view_file` to read each dependency file in order, then the target file.

5. Synthesize the information and answer the user's question. Do not dump raw markdown — provide a clear, concise answer.

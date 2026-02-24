# Knowledge Base Lookup Skill
- status: active
- type: agent_skill
- label: [agent, core]
<!-- content -->
This skill enables AI coding assistants to discover and read knowledge base documents with automatic dependency resolution. It uses `dependency_registry.json` as the catalog and the project's markdown files as the knowledge source.

## 1. How It Works
- status: active
- type: documentation
<!-- content -->
The knowledge base is organized as a dependency graph of Markdown files. Each file is registered in `dependency_registry.json` with:
- **`path`**: Location relative to project root (e.g. `content/skills/MCP_SKILL.md`)
- **`type`**: Category (`agent_skill`, `guideline`, `plan`, `log`, `documentation`, `task`)
- **`description`**: Two-sentence summary of the file's content
- **`dependencies`**: Map of files that must be read first

## 2. Lookup Procedure
- status: active
- type: agent_skill
<!-- content -->
When a user asks a question related to this project's knowledge base:

1. **Browse the catalog**: Read `dependency_registry.json`. Scan the `description` fields to identify which entries are relevant to the user's question.
2. **Resolve dependencies**: For the chosen entry, read its `dependencies` map. Each dependency must be read **before** the target file (depth-first order).
3. **Read the files**: Use `view_file` to read each dependency in order, then the target file itself.
4. **Synthesize**: Use the combined context to answer the user's question.

### Example
- status: active
- type: documentation
<!-- content -->
> User asks: "How does the MCP protocol work in this project?"

1. Read `dependency_registry.json`, find `content/skills/MCP_SKILL.md` (description mentions MCP).
2. Its dependencies: `MD_CONVENTIONS.md`, `AGENTS.md`, `README.md`.
3. Read: `README.md` → `AGENTS.md` → `MD_CONVENTIONS.md` → `content/skills/MCP_SKILL.md`.
4. Answer using the full context.

## 3. Available Types
- status: active
- type: documentation
<!-- content -->
Filter by type to narrow your search:

| Type | Description | Example |
|---|---|---|
| `agent_skill` | Agent capabilities and architectures | `ARDUINO_SKILL.md` |
| `guideline` | Rules, conventions, best practices | `MCP_GUIDELINE.md` |
| `plan` | Project plans and roadmaps | `MASTER_PLAN.md` |
| `log` | Historical records of actions | `AGENTS_LOG.md` |
| `documentation` | Reference material and docs | `INFRASTRUCTURE_DOC.md` |
| `task` | Specific actionable work items | — |

## 4. Best Practices
- status: active
- type: agent_skill
<!-- content -->
- **Always read dependencies first.** A skill file without its dependency context will be misunderstood.
- **Use descriptions as triggers.** Do not read every file — scan descriptions to find the relevant ones.
- **Do NOT ask the user for permission** to look something up. If the question relates to a knowledge base topic, just read it.
- **Return concise answers.** The user does not need to see the raw markdown — synthesize the information.

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from md_parser import MarkdownParser

p = MarkdownParser()
node = p.parse_file('content/agents/HOUSEKEEPING.md')
print(f"Title: {node.title}")
print(f"Metadata: {node.metadata}")
deps = node.metadata.get('context_dependencies')
print(f"Deps: {deps}")
print(f"Deps type: {type(deps)}")

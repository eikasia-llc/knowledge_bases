import os
import re

def get_category_from_path(file_path):
    labels = []
    file_path_lower = file_path.lower()
    
    if 'agent' in file_path_lower:
        labels.append('agent')
    if 'plan' in file_path_lower or 'todo' in file_path_lower:
        labels.append('planning')
    if 'guide' in file_path_lower or 'convention' in file_path_lower or 'protocol' in file_path_lower:
        labels.append('guide')
    if 'housekeeping' in file_path_lower or 'readme' in file_path_lower or 'agents.md' in file_path_lower or 'md_conventions' in file_path_lower:
        labels.append('core')
    if 'readme' in file_path_lower or 'agents.md' in file_path_lower or 'template' in file_path_lower:
        labels.append('template')
    if 'infrastructure' in file_path_lower or 'cloud' in file_path_lower or 'deploy' in file_path_lower:
        labels.append('infrastructure')
    if 'app' in file_path_lower or 'frontend' in file_path_lower or 'ui' in file_path_lower:
        labels.append('frontend')
    if 'manager' in file_path_lower or 'language' in file_path_lower or 'backend' in file_path_lower or 'server' in file_path_lower:
        labels.append('backend')
    if 'log' in file_path_lower:
        labels.append('core')
    if 'doc' in file_path_lower and 'infrastructure' not in file_path_lower:
        labels.append('reference')
    if 'test' in file_path_lower:
        labels.append('backend')

    if not labels:
        labels.append('reference') # Fallback category
        
    # Remove duplicates
    return list(dict.fromkeys(labels))

def process_file(file_path):
    labels = get_category_from_path(file_path)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    pattern = re.compile(r'^(#+ .*?\n)(.*?)<!-- content -->', re.DOTALL | re.MULTILINE)
    
    match = pattern.search(content)
    if not match:
        return False
        
    metadata_block = match.group(2)
    
    # Check if the FIRST metadata block has a label
    if '- label:' in metadata_block or 'label:' in metadata_block:
        return False
        
    label_str = "- label: [" + ", ".join(labels) + "]\n"
    new_metadata_block = metadata_block + label_str
    
    new_content = content[:match.start(2)] + new_metadata_block + content[match.end(2):]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Updated {file_path} with labels: {labels}")
    return True

if __name__ == '__main__':
    modified_count = 0
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'temprepo_cleaning' in root or 'repositories' in root or '.gemini' in root or '.pytest_cache' in root:
            continue
            
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                if process_file(file_path):
                    modified_count += 1
                    
    print(f"Process complete. Modified {modified_count} files.")

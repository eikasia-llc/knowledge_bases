import os
import re

def get_category_from_path(file_path):
    labels = []
    
    # Root files
    if file_path.startswith('./') and '/' not in file_path[2:]:
        if 'README.md' in file_path or 'AGENTS.md' in file_path:
            labels.extend(['template', 'core'])
        elif 'MD_CONVENTIONS.md' in file_path:
            labels.append('core')
        elif 'HOUSEKEEPING.md' in file_path:
            labels.extend(['guide', 'core'])
        elif 'TODOS.md' in file_path:
            labels.extend(['planning', 'draft'])
            
    # Content files
    elif 'content/agents/' in file_path:
        labels.append('agent')
    elif 'content/plans/' in file_path:
        labels.append('planning')
    elif 'content/guidelines/' in file_path:
        labels.append('guide')
    elif 'content/logs/' in file_path:
        labels.append('core')
    elif 'content/documentation/' in file_path:
        if 'INFRASTRUCTURE' in file_path:
            labels.append('infrastructure')
        else:
            labels.append('reference')
            
    # Remove duplicates
    labels = list(dict.fromkeys(labels))
    return labels

def process_file(file_path):
    labels = get_category_from_path(file_path)
    if not labels:
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Match the first header and its metadata block up to <!-- content -->
    # We look for ^#+\s+.*?\n((?:[-a-zA-Z0-9_]+:.*?\n)*)<!-- content -->
    
    pattern = re.compile(r'^(#+ .*?\n)(.*?)<!-- content -->', re.DOTALL | re.MULTILINE)
    match = pattern.search(content)
    
    if not match:
        return False
        
    header_line = match.group(1)
    metadata_block = match.group(2)
    
    if '- label:' in metadata_block or 'label:' in metadata_block:
        # Label already exists, skip
        return False
        
    # Create the new label line
    label_str = "- label: [" + ", ".join(labels) + "]\n"
    
    # Insert label at the end of the metadata block
    new_metadata_block = metadata_block + label_str
    
    new_content = content[:match.start(2)] + new_metadata_block + content[match.end(2):]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Updated {file_path} with labels: {labels}")
    return True

if __name__ == '__main__':
    modified_count = 0
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'temprepo_cleaning' in root or 'manager/cleaner' in root or '.gemini' in root or '.pytest_cache' in root:
            continue
            
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                if process_file(file_path):
                    modified_count += 1
                    
    print(f"Process complete. Modified {modified_count} files.")

#!/usr/bin/env python3
import os
import sys
import argparse
import difflib
import datetime

# Add manager/language to path for md_parser
current_dir = os.path.dirname(os.path.abspath(__file__))
manager_dir = os.path.dirname(current_dir)
language_dir = os.path.join(manager_dir, 'language')
sys.path.append(language_dir)

from md_parser import MarkdownParser, Node

# Function to find target file in content directory
def find_target_file(filename, content_root):
    for root, dirs, files in os.walk(content_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

def merge_metadata(target_meta, source_meta):
    """
    Merge metadata dictionaries. 
    - Lists are unioned.
    - Single values: Source overrides Target if different (as Source is 'newer').
    """
    merged = target_meta.copy()
    
    for key, value in source_meta.items():
        if key in merged:
            target_val = merged[key]
            if isinstance(target_val, list) and isinstance(value, list):
                # Union of lists, preserving order
                new_list = target_val[:]
                for item in value:
                    if item not in new_list:
                        new_list.append(item)
                merged[key] = new_list
            else:
                # Overwrite or Add
                merged[key] = value
        else:
            merged[key] = value
            
    return merged

def smart_merge_nodes(target_node, source_node):
    """
    Recursively merge source_node into target_node.
    """
    # 1. Merge Metadata
    target_node.metadata = merge_metadata(target_node.metadata, source_node.metadata)
    
    # 2. Merge Content
    t_content = target_node.content.strip() if target_node.content else ""
    s_content = source_node.content.strip() if source_node.content else ""
    
    if t_content == s_content:
        pass # Identical
    elif t_content and t_content in s_content:
        # Target is substring of Source -> Upgrade to Source
        target_node.content = source_node.content
    elif s_content and s_content in t_content:
        # Source is substring of Target -> Keep Target (it has more info)
        pass 
    else:
        # Conflict / Divergence -> Append Source
        if s_content: # Only if source has content
            separator = "\n\n<!-- MERGED FROM NEWER VERSION -->\n\n"
            target_node.content = t_content + separator + s_content

    # 3. Merge Children
    # Map target children by Title for easy lookup
    target_children_map = {child.title: child for child in target_node.children}
    
    for source_child in source_node.children:
        if source_child.title in target_children_map:
            # Match found -> Recurse
            smart_merge_nodes(target_children_map[source_child.title], source_child)
        else:
            # New child -> Append
            # We need to clone it to be safe, but Python object ref is fine here as source is discarded
            target_node.children.append(source_child)

    return target_node

def process_repository(repo_dir, content_dir, dry_run=False):
    print(f"Scanning repository: {repo_dir}")
    parser = MarkdownParser()
    
    updates = []
    new_files = []
    
    for root, dirs, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith('.md'):
                continue
                
            source_path = os.path.join(root, file)
            target_path = find_target_file(file, content_dir)
            
            if target_path:
                # Merge needed
                print(f"Comparing {file} ...")
                try:
                    target_root = parser.parse_file(target_path)
                    source_root = parser.parse_file(source_path)
                    
                    # Merge Logic
                    smart_merge_nodes(target_root, source_root)
                    
                    # Check for changes (naive check: compare serialized output)
                    new_content = target_root.to_markdown()
                    with open(target_path, 'r') as f:
                        old_content = f.read()
                        
                    if new_content != old_content:
                        updates.append((target_path, new_content))
                        print(f"  -> Changes detected in {file}")
                    else:
                        print(f"  -> No changes for {file}")
                        
                except Exception as e:
                    print(f"Error merging {file}: {e}")
            else:
                # New File
                print(f"New file found: {file}")
                # Determine where to put it. 
                # For now, put it in content/imported/ OR just report it.
                # The user asked to "put them in the \repositories folder" which is done.
                # This script is for merging existing ones. 
                # If it's new, we should probably copy it to a smart location, but for safety let's just log it.
                # Or putting it in content/incoming?
                # Let's put it in content/ if it doesn't exist, maybe in root or 'agents'? 
                # Hard to guess category. Let's dump to content/incoming if it exists, or just root content/
                
                # Logic: If source was in 'agents/', put in 'content/agents/'? 
                # source path: .../repositories/mcmp_chatbot/AGENTS.md
                # We can try to preserve relative structure?
                
                # For this iteration: List new files.
                new_files.append(source_path)

    # Execute Updates
    if not dry_run:
        print("\nApplying Changes...")
        for path, content in updates:
            with open(path, 'w') as f:
                f.write(content)
            print(f"Updated {path}")
            
        if new_files:
            print("\nNew Files Found (Not automatically moved to content/ to avoid clutter):")
            for f in new_files:
                print(f"- {f}")
            print("You can move them manually.")
    else:
        print("\nDry Run - No changes applied.")
        print(f"Would update {len(updates)} files.")
        print(f"Would find {len(new_files)} new files.")

def main():
    parser = argparse.ArgumentParser(description="Compare and Smart Merge repositories into Content.")
    parser.add_argument("--repo_dir", default=os.path.join(current_dir, "repositories"), help="Directory containing ingested repos")
    parser.add_argument("--content_dir", default=os.path.join(manager_dir, "../content"), help="Target content directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply changes, just show what would happen")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.repo_dir):
        print(f"Repository directory not found: {args.repo_dir}")
        sys.exit(1)
        
    # We Iterate over each repo folder in repositories/
    for item in os.listdir(args.repo_dir):
        item_path = os.path.join(args.repo_dir, item)
        if os.path.isdir(item_path):
            process_repository(item_path, args.content_dir, args.dry_run)

if __name__ == "__main__":
    main()

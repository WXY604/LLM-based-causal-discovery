'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-05-28 20:20:18
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-05-28 20:26:13
FilePath: /IJCNN_提交版/prompt_design/prompt_generation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import os
from pathlib import Path

def process_json_files(input_dir="prompt_design/description", output_base_dir="prompt_design/prompt"):
    # Create output directory if it doesn't exist
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each JSON file in the input directory
    for json_file in Path(input_dir).glob("*.json"):
        print(f"\nProcessing file: {json_file}")
        
        with open(json_file, 'r') as f:
            config = json.load(f)
        
        # Generate variable definitions text
        variable_definitions = "\n".join(
            f"{var}: {desc}" for var, desc in config['variable_definitions'].items()
        )
        
        for current_var in config['variable_definitions'].keys():
            # Format the query template
            query_text = config['query_template']\
                .replace("{{variable_definitions}}", variable_definitions)\
                .replace("{{current_variable}}", current_var)
            
            # Create output filename based on input JSON filename and current variable
            output_filename = f"{json_file.stem}_{current_var}.txt"
            output_dictionary = Path(output_base_dir) / json_file.stem
            output_path = Path(output_base_dir) / json_file.stem / output_filename
            os.makedirs(output_dictionary, exist_ok=True)

            # Write to file
            with open(output_path, 'w') as f:
                f.write(query_text)
            print(f"  Generated: {output_filename}")

if __name__ == "__main__":
    input_dir="prompt_design/description"
    output_dir="prompt_design/prompt"
    process_json_files(input_dir,output_dir)
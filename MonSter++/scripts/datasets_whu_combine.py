#!/usr/bin/env python3

"""
WHU-Stereo Fixed Concatenation Script
This script uses proper concatenation method for multi-part zip files
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def create_directories():
    """Create the necessary directory structure"""
    print("Creating directory structure...")
    
    directories = [
        'datasets/whu',
        'datasets/whu/training/left',
        'datasets/whu/training/right',
        'datasets/whu/training/disp',
        'datasets/whu/testing/left',
        'datasets/whu/testing/right'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")

def combine_zip_parts_fixed(parts_dir):
    """Combine multi-part zip files with proper concatenation"""
    print("Combining multi-part zip files with proper concatenation...")
    
    try:
        # Find all zip parts
        zip_parts = []
        for file in os.listdir(parts_dir):
            if file.startswith('WHU-Stereo dataset') and (file.endswith('.zip') or file.endswith('.z01') or file.endswith('.z02') or file.endswith('.z03') or file.endswith('.z04') or file.endswith('.z05') or file.endswith('.z06') or file.endswith('.z07') or file.endswith('.z08') or file.endswith('.z09') or file.endswith('.z10') or file.endswith('.z11') or file.endswith('.z12') or file.endswith('.z13')):
                zip_parts.append(file)
        
        # Sort in the correct order for multi-part zip files
        def sort_key(filename):
            if filename.endswith('.zip'):
                return 1
            elif filename.endswith('.z01'):
                return 2
            elif filename.endswith('.z02'):
                return 3
            elif filename.endswith('.z03'):
                return 4
            elif filename.endswith('.z04'):
                return 5
            elif filename.endswith('.z05'):
                return 6
            elif filename.endswith('.z06'):
                return 7
            elif filename.endswith('.z07'):
                return 8
            elif filename.endswith('.z08'):
                return 9
            elif filename.endswith('.z09'):
                return 10
            elif filename.endswith('.z10'):
                return 11
            elif filename.endswith('.z11'):
                return 12
            elif filename.endswith('.z12'):
                return 13
            elif filename.endswith('.z13'):
                return 14
            else:
                return 15
        
        zip_parts.sort(key=sort_key)
        print(f"Found {len(zip_parts)} zip parts in correct order: {zip_parts}")
        
        if len(zip_parts) == 0:
            print("✗ No zip parts found")
            return False
        
        # Combine the files with proper concatenation
        output_file = os.path.join(parts_dir, 'WHU-Stereo-complete.zip')
        
        with open(output_file, 'wb') as outfile:
            for part in zip_parts:
                part_path = os.path.join(parts_dir, part)
                print(f"Adding part: {part}")
                
                # Read the entire file into memory and write it
                with open(part_path, 'rb') as infile:
                    data = infile.read()
                    outfile.write(data)
                    
                    # Ensure we flush the data
                    outfile.flush()
                    os.fsync(outfile.fileno())
        
        print(f"✓ Combined zip file created: {output_file}")
        
        # Verify the combined file size
        total_size = sum(os.path.getsize(os.path.join(parts_dir, part)) for part in zip_parts)
        combined_size = os.path.getsize(output_file)
        print(f"Expected size: {total_size / (1024*1024):.1f} MB")
        print(f"Actual size: {combined_size / (1024*1024):.1f} MB")
        
        if total_size != combined_size:
            print("⚠️ Warning: Combined file size doesn't match expected size")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Error combining zip parts: {e}")
        return False

def extract_with_system_tools(zip_path):
    """Extract using system tools that handle multi-part archives better"""
    print("Extracting with system tools...")
    
    try:
        # Try using cat to combine and pipe to unzip
        print("Trying cat + unzip method...")
        
        # First, let's try to extract directly with unzip
        import subprocess
        result = subprocess.run(['unzip', '-o', zip_path, '-d', 'datasets/whu/'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset extracted successfully with unzip")
            return True
        else:
            print(f"Unzip failed: {result.stderr}")
            
            # Try with zip command to fix first
            print("Trying zip fix method...")
            result = subprocess.run(['zip', '-F', zip_path, '--out', zip_path + '.fixed'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                print("✓ Zip file fixed, trying extraction...")
                fixed_zip = zip_path + '.fixed'
                result = subprocess.run(['unzip', '-o', fixed_zip, '-d', 'datasets/whu/'], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if result.returncode == 0:
                    print("✓ Dataset extracted successfully after fixing")
                    # Clean up fixed file
                    if os.path.exists(fixed_zip):
                        os.remove(fixed_zip)
                    return True
                else:
                    print(f"Extraction after fix failed: {result.stderr}")
            
            # Try 7zip if available
            print("Trying 7zip...")
            result = subprocess.run(['7z', 'x', zip_path, '-odatasets/whu/', '-y'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                print("✓ Dataset extracted successfully with 7zip")
                return True
            else:
                print(f"7zip failed: {result.stderr}")
        
        return False
        
    except Exception as e:
        print(f"✗ Error extracting with system tools: {e}")
        return False

def organize_dataset():
    """Organize the extracted dataset"""
    print("Organizing files...")
    
    try:
        # Move training data
        # Check both possible paths for the data
        possible_paths = [
            'datasets/whu/with ground truth/train',
            'datasets/whu/experimental data/with ground truth/train',
            'datasets/whu/train'
        ]
        
        train_path = None
        for path in possible_paths:
            if os.path.exists(path):
                train_path = path
                break
        
        if train_path:
            print(f"Moving training data from: {train_path}")
            
            # Move left images
            left_src = os.path.join(train_path, 'left')
            if os.path.exists(left_src):
                for file in os.listdir(left_src):
                    if file.endswith(('.png', '.tiff', '.tif')):
                        shutil.move(
                            os.path.join(left_src, file),
                            f'datasets/whu/training/left/{file}'
                        )
            
            # Move right images
            right_src = os.path.join(train_path, 'right')
            if os.path.exists(right_src):
                for file in os.listdir(right_src):
                    if file.endswith(('.png', '.tiff', '.tif')):
                        shutil.move(
                            os.path.join(right_src, file),
                            f'datasets/whu/training/right/{file}'
                        )
            
            # Move disparity maps
            disp_src = os.path.join(train_path, 'disp')
            if os.path.exists(disp_src):
                for file in os.listdir(disp_src):
                    if file.endswith(('.png', '.tiff', '.tif')):
                        shutil.move(
                            os.path.join(disp_src, file),
                            f'datasets/whu/training/disp/{file}'
                        )
        
        # Move testing data
        test_paths = [
            'datasets/whu/with ground truth/test',
            'datasets/whu/experimental data/with ground truth/test',
            'datasets/whu/test'
        ]
        
        test_path = None
        for path in test_paths:
            if os.path.exists(path):
                test_path = path
                break
        
        if test_path:
            print(f"Moving testing data from: {test_path}")
            
            # Move left images
            left_src = os.path.join(test_path, 'left')
            if os.path.exists(left_src):
                for file in os.listdir(left_src):
                    if file.endswith(('.png', '.tiff', '.tif')):
                        shutil.move(
                            os.path.join(left_src, file),
                            f'datasets/whu/testing/left/{file}'
                        )
            
            # Move right images
            right_src = os.path.join(test_path, 'right')
            if os.path.exists(right_src):
                for file in os.listdir(right_src):
                    if file.endswith(('.png', '.tiff', '.tif')):
                        shutil.move(
                            os.path.join(right_src, file),
                            f'datasets/whu/testing/right/{file}'
                        )
        
        # Clean up extracted files
        print("Cleaning up...")
        cleanup_paths = [
            'datasets/whu/with ground truth',
            'datasets/whu/without ground truth',
            'datasets/whu/experimental data'
        ]
        
        for path in cleanup_paths:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Removed: {path}")
        
        print("✓ Dataset organization complete")
        return True
        
    except Exception as e:
        print(f"✗ Error organizing dataset: {e}")
        return False

def verify_dataset():
    """Verify the dataset structure"""
    print("Verifying dataset structure...")
    
    try:
        # Check training structure
        if (os.path.exists('datasets/whu/training/left') and 
            os.path.exists('datasets/whu/training/right') and 
            os.path.exists('datasets/whu/training/disp')):
            
            left_count = len([f for f in os.listdir('datasets/whu/training/left') if f.endswith(('.png', '.tiff', '.tif'))])
            right_count = len([f for f in os.listdir('datasets/whu/training/right') if f.endswith(('.png', '.tiff', '.tif'))])
            disp_count = len([f for f in os.listdir('datasets/whu/training/disp') if f.endswith(('.png', '.tiff', '.tif'))])
            
            print(f"✓ WHU stereo training structure is correct")
            print(f"  - Left images: {left_count} files")
            print(f"  - Right images: {right_count} files")
            print(f"  - Disparity maps: {disp_count} files")
        else:
            print("✗ WHU stereo training structure is incorrect")
            return False
        
        # Check testing structure
        if (os.path.exists('datasets/whu/testing/left') and 
            os.path.exists('datasets/whu/testing/right')):
            
            left_count = len([f for f in os.listdir('datasets/whu/testing/left') if f.endswith(('.png', '.tiff', '.tif'))])
            right_count = len([f for f in os.listdir('datasets/whu/testing/right') if f.endswith(('.png', '.tiff', '.tif'))])
            
            print(f"✓ WHU stereo testing structure is correct")
            print(f"  - Left images: {left_count} files")
            print(f"  - Right images: {right_count} files")
        else:
            print("✗ WHU stereo testing structure is incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying dataset: {e}")
        return False

def main():
    """Main function"""
    print("=== WHU-Stereo Fixed Concatenation Script ===")
    print("This script uses proper concatenation method for multi-part zip files")
    print("")
    
    # Create directories
    create_directories()
    
    # Check if parts directory exists
    parts_dir = 'datasets/whu/parts'
    if not os.path.exists(parts_dir):
        print(f"✗ Parts directory not found: {parts_dir}")
        print("Please download the WHU-Stereo parts first")
        sys.exit(1)
    
    # Combine the zip parts with fixed concatenation
    combined_zip = combine_zip_parts_fixed(parts_dir)
    if not combined_zip:
        print("✗ Failed to combine zip parts")
        sys.exit(1)
    
    # Extract with system tools
    if extract_with_system_tools(combined_zip):
        if organize_dataset():
            if verify_dataset():
                print("\n=== Setup Complete ===")
                print("WHU-Stereo dataset has been successfully extracted and organized!")
                return
            else:
                print("✗ Dataset verification failed")
                sys.exit(1)
        else:
            print("✗ Dataset organization failed")
            sys.exit(1)
    else:
        print("✗ Dataset extraction failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 
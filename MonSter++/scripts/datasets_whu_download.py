#!/usr/bin/env python3

"""
WHU-Stereo Complete Dataset Download Script (New Link)
This script downloads the complete WHU-Stereo dataset with all 13 parts
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
import pickle
import io

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# New WHU-Stereo dataset folder ID
WHU_FOLDER_ID = '1EsOgmyhbQYQYn7ApoEFtuZkBZgygBfcV'

def check_dependencies():
    """Check if required packages are available"""
    try:
        import googleapiclient
        import google_auth_oauthlib
        print("✓ Google Drive API dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        sys.exit(1)

def authenticate_google_drive():
    """Authenticate with Google Drive API"""
    creds = None
    
    # Check if token.pickle exists (saved credentials)
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                print("✗ Authentication token expired. Please re-authenticate.")
                if os.path.exists('token.pickle'):
                    os.remove('token.pickle')
                creds = None
        
        if not creds:
            print("Setting up Google Drive authentication...")
            print("This will open a browser window for authentication.")
            print("Please follow the instructions to authorize this application.")
            
            # Check if credentials.json exists
            if not os.path.exists('credentials.json'):
                print("✗ credentials.json not found!")
                print("Please download credentials.json from Google Cloud Console:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project or select existing one")
                print("3. Enable Google Drive API")
                print("4. Create credentials (OAuth 2.0 Client ID)")
                print("5. Download credentials.json and place it in this directory")
                sys.exit(1)
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

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

def find_whu_files_in_drive(service, folder_id):
    """Find all WHU-Stereo files in the Google Drive folder"""
    print("Searching for WHU-Stereo files in Google Drive...")
    
    try:
        # List files in the folder
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id,name,size)"
        ).execute()
        
        files = results.get('files', [])
        
        whu_files = []
        for file in files:
            if 'WHU-Stereo dataset' in file['name']:
                whu_files.append(file)
                print(f"Found: {file['name']} (ID: {file['id']})")
        
        return whu_files
        
    except Exception as e:
        print(f"✗ Error searching for files: {e}")
        return []

def download_file_from_drive(service, file_id, destination):
    """Download a file from Google Drive"""
    try:
        # Get file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata.get('name', 'Unknown file')
        
        # Handle file size (may not be available for all files)
        file_size = file_metadata.get('size')
        if file_size:
            file_size = int(file_size)
            print(f"Downloading: {file_name}")
            print(f"File size: {file_size / (1024*1024):.1f} MB")
        else:
            print(f"Downloading: {file_name}")
            print("File size: Unknown (large file)")
        
        # Create a file stream for downloading
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        chunk_count = 0
        while not done:
            status, done = downloader.next_chunk()
            chunk_count += 1
            if status:
                progress = int(status.progress() * 100)
                print(f"Download progress: {progress}%")
            elif chunk_count % 10 == 0:  # Show progress every 10 chunks even without status
                print(f"Downloading... (chunk {chunk_count})")
        
        # Save the file
        fh.seek(0)
        with open(destination, 'wb') as f:
            shutil.copyfileobj(fh, f)
        
        # Get actual file size
        actual_size = os.path.getsize(destination)
        print(f"✓ Successfully downloaded: {file_name}")
        print(f"Actual file size: {actual_size / (1024*1024):.1f} MB")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        return False

def combine_zip_parts(parts_dir):
    """Combine multi-part zip files"""
    print("Combining multi-part zip files...")
    
    try:
        # Find all zip parts
        zip_parts = []
        for file in os.listdir(parts_dir):
            if file.startswith('WHU-Stereo dataset') and (file.endswith('.zip') or file.endswith('.z01') or file.endswith('.z02') or file.endswith('.z03') or file.endswith('.z04') or file.endswith('.z05') or file.endswith('.z06') or file.endswith('.z07') or file.endswith('.z08') or file.endswith('.z09') or file.endswith('.z10') or file.endswith('.z11') or file.endswith('.z12') or file.endswith('.z13')):
                zip_parts.append(file)
        
        # Sort in the correct order for multi-part zip files
        # The .zip file should be first, followed by .z01, .z02, etc.
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
        
        # Combine the files
        output_file = os.path.join(parts_dir, 'WHU-Stereo-complete.zip')
        
        with open(output_file, 'wb') as outfile:
            for part in zip_parts:
                part_path = os.path.join(parts_dir, part)
                print(f"Adding part: {part}")
                with open(part_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        
        print(f"✓ Combined zip file created: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"✗ Error combining zip parts: {e}")
        return False

def extract_and_organize(zip_path):
    """Extract the zip file and organize the dataset"""
    print("Extracting and organizing dataset...")
    
    try:
        # Try to extract with different methods
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('datasets/whu/')
            print("✓ Dataset extracted successfully with standard zip extraction")
        except zipfile.BadZipFile as e:
            print(f"Standard zip extraction failed: {e}")
            print("Trying alternative extraction method...")
            
            # Try using system unzip command
            import subprocess
            result = subprocess.run(['unzip', '-o', zip_path, '-d', 'datasets/whu/'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print("✓ Dataset extracted successfully with system unzip")
            else:
                print(f"System unzip failed: {result.stderr}")
                raise Exception("All extraction methods failed")
        
        # Organize the files
        print("Organizing files...")
        
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
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed: {zip_path}")
        
        print("✓ Dataset organization complete")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting dataset: {e}")
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
    print("=== WHU-Stereo Complete Dataset Download Script (New Link) ===")
    print("This script downloads the complete WHU-Stereo dataset with all 13 parts")
    print("")
    
    # Check dependencies
    check_dependencies()
    
    # Create directories
    create_directories()
    
    # Authenticate with Google Drive
    print("Authenticating with Google Drive...")
    creds = authenticate_google_drive()
    
    # Build the service
    service = build('drive', 'v3', credentials=creds)
    
    # Find all WHU files in the folder
    whu_files = find_whu_files_in_drive(service, WHU_FOLDER_ID)
    
    if not whu_files:
        print("✗ No WHU-Stereo files found in the folder")
        print("Please check the Google Drive folder manually")
        sys.exit(1)
    
    # Download all files in the correct order
    print("Downloading all WHU-Stereo files in correct order...")
    parts_dir = 'datasets/whu/parts'
    os.makedirs(parts_dir, exist_ok=True)
    
    # Sort files in the correct order for multi-part zip
    # The .zip file should be first, followed by .z01, .z02, etc.
    def sort_key(file_info):
        filename = file_info['name']
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
    
    whu_files.sort(key=sort_key)
    print(f"Download order: {[f['name'] for f in whu_files]}")
    
    download_success = True
    for file_info in whu_files:
        file_id = file_info['id']
        file_name = file_info['name']
        destination = os.path.join(parts_dir, file_name)
        
        if not download_file_from_drive(service, file_id, destination):
            download_success = False
            break
    
    if not download_success:
        print("✗ Failed to download all files")
        sys.exit(1)
    
    # Combine the zip parts
    combined_zip = combine_zip_parts(parts_dir)
    if not combined_zip:
        print("✗ Failed to combine zip parts")
        sys.exit(1)
    
    # Extract and organize
    if extract_and_organize(combined_zip):
        # Verify the dataset
        if verify_dataset():
            print("")
            print("=== Setup Complete ===")
            print("WHU-Stereo dataset has been successfully downloaded and organized!")
            print("")
            print("Dataset information:")
            print("- Source: WHU-Stereo: A Challenging Benchmark for Stereo Matching of High-Resolution Satellite Images")
            print("- Paper: IEEE Transactions on Geoscience and Remote Sensing, 2023")
            print("- Cities: Wuhan, Hengyang, Shaoguan, Kunming, Yingde, and Qichun")
            print("- Resolution: High-resolution satellite imagery")
            print("")
            print("You can now use the dataset with:")
            print("- Training: python train_whu.py")
            print("- Evaluation: make evaluate_whu")
            print("")
        else:
            print("✗ Dataset verification failed")
            sys.exit(1)
    else:
        print("✗ Dataset extraction failed")
        sys.exit(1)

if __name__ == '__main__':
    main() 
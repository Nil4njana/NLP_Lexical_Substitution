import os
import sys
import tarfile

def main():
    print("Installing 'gdown' for Google Drive download...")
    os.system(sys.executable + " -m pip install gdown")
    import gdown
    
    file_id = '13OzoeD_pWF4XnUfZYYYitTIV25FUC3Z4'
    url = f'https://drive.google.com/uc?id={file_id}'
    out_file = 'similarity_new_bert4.tar.xz'
    
    print("\nDownloading 1.5GB model from Google Drive...")
    if not os.path.exists(out_file):
        gdown.download(url, out_file, quiet=False)
    else:
        print("Archive already exists, skipping download.")
        
    extract_dir = os.path.join("checkpoint", "similarity_new_bert")
    os.makedirs(extract_dir, exist_ok=True)
    
    print("\nExtracting massive tar.xz archive (this will take a minute or two)....")
    try:
        with tarfile.open(out_file, "r:xz") as tar:
            tar.extractall(path=extract_dir)
        print("\nSUCCESS! Extraction complete.")
    except Exception as e:
        print("\nERROR during extraction:", str(e))
        sys.exit(1)
        
    print(f"Weights are now successfully located in: {extract_dir}")
    print("You can now safely delete the 1.5GB 'similarity_new_bert4.tar.xz' file.")

if __name__ == '__main__':
    main()

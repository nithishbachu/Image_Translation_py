from PIL import Image
import os

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_size(file):
    """Validate image file size is under MAX_FILE_SIZE_MB"""
    # Get file size in MB
    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)  # Reset file pointer
    
    return size_mb <= MAX_FILE_SIZE_MB

def process_upload(file):
    """Process and validate uploaded image"""
    if not file:
        raise ValueError("No file provided")
    
    if not allowed_file(file.filename):
        raise ValueError(f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    if not validate_image_size(file):
        raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
    
    return Image.open(file)
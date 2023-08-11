# from google.colab import drive
# drive.mount('/content/drive')

#WORKING AND EDITABLE

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import os

def create_google_doc(string):
    # connect creds
    gauth = GoogleAuth()
    link = ['https://www.googleapis.com/auth/drive']
    cred = ServiceAccountCredentials.from_json_keyfile_name('C:\\Users\\veena\\Desktop\\GeorgiaTech\\NowNotes\\src\\clientSecrets.json', link)
    gauth.credentials = cred

    drive = GoogleDrive(gauth)

    # create doc
    doc_metadata = {
        'title': 'My Document',
        'mimeType': 'application/vnd.google-apps.document'
    }
    doc = drive.CreateFile(doc_metadata)
    doc.Upload()

    # anyone can edit
    permissions = {
        'type': 'anyone',
        'role': 'writer',
    }

    doc.InsertPermission(permissions)
    content = string
    doc.SetContentString(content)
    doc.Upload()

    # Retrieve the link to the Google Doc
    url = doc['alternateLink']
    print("Google Doc created successfully!")
    print("You can access the document at the following link:")
    print(url)
    return url

def read_file_content(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return "File not found."



def delete_uploaded_images():
    # Define the path to the "uploads" folder
    uploads_folder = 'uploads'

    # Get a list of all files in the uploads folder
    files = os.listdir(uploads_folder)

    # Iterate through the files and delete each one
    for file in files:
        file_path = os.path.join(uploads_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

delete_uploaded_images()

def delete_file_content(filename):
    try:
        with open(filename, 'w'):
            pass  # This just opens and immediately closes the file
        print(f"Content deleted from {filename}")
    except Exception as e:
        print(f"An error occurred while deleting content: {e}")

def delete_uploaded_images():
    # Define the path to the "uploads" folder
    uploads_folder = 'uploads'

    # Get a list of all files in the uploads folder
    files = os.listdir(uploads_folder)

    # Iterate through the files and delete each one
    for file in files:
        file_path = os.path.join(uploads_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def delete_toProcess_images():
    # Define the path to the "uploads" folder
    uploads_folder = 'toProcess'

    # Get a list of all files in the uploads folder
    files = os.listdir(uploads_folder)

    # Iterate through the files and delete each one
    for file in files:
        file_path = os.path.join(uploads_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# result_string = ""
# filename = 'paragraph.txt'
# text = read_file_content('paragraph.txt')
# print(text)
# doc_url = create_google_doc(text)
# webbrowser.open(doc_url)
# delete_file_content(filename)
# delete_uploaded_images()
# delete_toProcess_images()



from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os, time, requests, zipfile

# Load the endpoint and keys of your resource
load_dotenv()
training_endpoint = os.getenv('TRAINING_ENDPOINT')
training_key = os.getenv('TRAINING_KEY')
project_id = os.getenv('PROJECT_ID')
iteration_id = os.getenv('ITERATION_ID')

# Authenticate the client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(training_endpoint, credentials)

platform = "TensorFlow"
flavor = "TensorFlow"
export = trainer.export_iteration(project_id, iteration_id, platform, flavor, raw=False)

while (export.status == "Exporting"):
    print("Waiting 10 seconds...")
    time.sleep(10)
    exports = trainer.get_exports(project_id, iteration_id)
    # Find the export for this iteration  
    for e in exports:
        if e.platform == export.platform and e.flavor == export.flavor:
            export = e
            break
    print("Export status is: ", export.status)

if export.status == "Done":
    # Download the model
    export_file = requests.get(export.download_uri)
    with open("export.zip", "wb") as file:
        file.write(export_file.content)

# Unzip the downloaded export
if not os.path.exists("./model"):
    os.mkdir("./model");
zip_ref = zipfile.ZipFile("export.zip", 'r')
zip_ref.extractall("./model")
zip_ref.close()
print("Data extracted in: ./model")
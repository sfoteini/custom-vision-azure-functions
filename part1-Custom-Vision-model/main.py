from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
from dotenv import load_dotenv
import os, time

# Load the endpoint and keys of your resource
load_dotenv()
training_endpoint = os.getenv('TRAINING_ENDPOINT')
training_key = os.getenv('TRAINING_KEY')
prediction_endpoint = os.getenv('PREDICTION_ENDPOINT')
prediction_key = os.getenv('PREDICTION_KEY')
prediction_resource_id = os.getenv('PREDICTION_RESOURCE_ID')

# Authenticate the client
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(training_endpoint, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)

# Find the domain id
classification_domain = next(domain for domain in trainer.get_domains() if domain.type == "Classification" and domain.name == "General (compact)")

# Create a new project
publish_iteration_name = "Iteration1"
project_name = "Cats and Dogs classifier"
project_description = "A Custom Vision project to detect cats and dogs"
domain_id = classification_domain.id
classification_type = "Multiclass"
print ("Creating project...")
project = trainer.create_project(project_name, project_description, domain_id, classification_type)

# Create tags for cats and dogs
cat_tag = trainer.create_tag(project.id, "Cat")
dog_tag = trainer.create_tag(project.id, "Dog")

# Upload and tag images
images_folder = os.path.join(os.path.dirname(__file__), "images", "Train")
tags_folder_names = [ "Cat", "Dog" ]

print("Adding images...")

for tag_num in range(0, 2):
    if tag_num == 0:
        tag = cat_tag
    else:
        tag = dog_tag
    for batch_num in range(0, 2):
        image_list = []
        for image_num in range(1, 61):
            file_name = f"{tags_folder_names[tag_num]} ({60*batch_num + image_num}).jpg"
            with open(os.path.join(images_folder, tags_folder_names[tag_num], file_name), "rb") as image_contents:
                image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[tag.id]))

        upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            exit(-1)
    print(f"{tags_folder_names[tag_num]} Uploaded")

# Training
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 20 seconds...")
    time.sleep(20)

# Get iteration performance information
threshold = 0.5
iter_performance_info = trainer.get_iteration_performance(project.id, iteration.id, threshold)
print("Iteration Performance:")
print(f"\tPrecision: {iter_performance_info.precision*100 :.2f}%\n"
      f"\tRecall: {iter_performance_info.recall*100 :.2f}%\n"
      f"\tAverage Precision: {iter_performance_info.average_precision*100 :.2f}%")

print("Performance per tag:")
for item in iter_performance_info.per_tag_performance:
    print(f"* {item.name}:")
    print(f"\tPrecision: {item.precision*100 :.2f}%\n"
          f"\tRecall: {item.recall*100 :.2f}%\n"
          f"\tAverage Precision: {item.average_precision*100 :.2f}%")

# Quick test
test_images_folder_path = os.path.join(os.path.dirname(__file__), "images", "Test")
test_image_filename = "4.jpg"

print("Quick test a local image...")
with open(os.path.join(test_images_folder_path, test_image_filename), "rb") as image_contents:
    quick_test_results = trainer.quick_test_image(project.id, image_contents.read(), iteration_id=iteration.id)
    # Display the results
    print(f"Quick Test results for image {test_image_filename}:")
    for prediction in quick_test_results.predictions:
        print(f"\t{prediction.tag_name}: {prediction.probability*100 :.2f}%")

# Publish the current iteration
print("Publishing the current iteration...")
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Iteration published!")

# Test - Make a prediction
print("Testing the prediction endpoint...")
for img_num in range(1,9):
    test_image_filename = str(img_num) + ".jpg"
    with open(os.path.join(test_images_folder_path, test_image_filename), "rb") as image_contents:
        results = predictor.classify_image(project.id, publish_iteration_name, image_contents.read())

        # Display the results
        print(f"Testing image {test_image_filename}...")
        for prediction in results.predictions:
            print(f"\t{prediction.tag_name}: {prediction.probability*100 :.2f}%")
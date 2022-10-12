import logging
import azure.functions as func

# Import helper code for running the TensorFlow model
from .predict import predict_image_from_url

def main(req: func.HttpRequest) -> func.HttpResponse:
    image_url = req.params.get('img')
    if not image_url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            image_url = req_body.get('img')
    logging.info('Image URL received: ' + image_url)
    prediction = predict_image_from_url(image_url)

    return func.HttpResponse(f"Classified as {prediction['label']} with probability {prediction['probability']*100 :.3f}%")
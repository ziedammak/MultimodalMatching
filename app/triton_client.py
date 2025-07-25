import tritonclient.grpc as grpcclient
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class TritonClient:
    def __init__(self, url="localhost:8001"):
        self.client = None
        self.url = url
        self.connect()
        
    def connect(self, retries=5, delay=2):
        for i in range(retries):
            try:
                self.client = grpcclient.InferenceServerClient(url=self.url)
                logger.info(f"Connected to Triton server at {self.url}")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {i+1}/{retries} failed: {str(e)}")
                if i < retries - 1:
                    time.sleep(delay)
        raise ConnectionError(f"Could not connect to Triton server at {self.url} after {retries} attempts")
    
    def infer(self, model_name, input_data, input_name, output_name):
        # Create input object
        inputs = [grpcclient.InferInput(input_name, input_data.shape, 
                   "FP32" if input_data.dtype == np.float32 else "INT32")]
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output object
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        
        try:
            # Perform inference
            response = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            return response.as_numpy(output_name)
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            # Try to reconnect
            self.connect()
            return self.infer(model_name, input_data, input_name, output_name)
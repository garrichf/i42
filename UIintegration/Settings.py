class Settings:

    def __init__(self):
       self.model_choice="YOLOv8" # Default model 
       self.confidence_threshold=0.5 
       self.use_webcam=True # Default is to use webcam 

    def set_model_choice(self ,choice):
       self.model_choice=choice 

    def set_confidence_threshold(self ,threshold):
       self.confidence_threshold=threshold 

    def set_data_source(self ,use_webcam):
       self.use_webcam=use_webcam 

    def get_model_choice(self):
       return self.model_choice 

    def get_confidence_threshold(self):
       return self.confidence_threshold 

    def get_data_source(self):
       return self.use_webcam 
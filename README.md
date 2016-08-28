# A model to predict the time required to deliver a package.

## File Descriptions
- corpus_handel.py : Data handeler to manage text data and convert into numerical form.
- readjson.py : To read raw json data and perform feature engineering. Output will be the required variable as destinationbranch', 'originbranch' , 'producttype' and 'overall_time' in an excel file.
- train_model.py : Implementation of Neural Network regression model to perform time prediction.
- run_model.py : To execute the generated model.
- prediction_api.py : Deployement of model as REST service.


## Execution Steps:
- Performing Feature Engineering and pre-processing.
  - python readjson.py
- Train model
  - python train_model.py
- Execute model
  - python run_model.py




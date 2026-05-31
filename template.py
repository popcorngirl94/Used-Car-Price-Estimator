## Template.py
from pathlib import Path
print("Script is starting...")
# Directories to create
directories=[
    #source
    "src/components",
    "src/configuration",
    "src/cloud_storage",
    "src/data_access",
    "src/constants",
    "src/entity",
    "src/exception",
    "src/logger",
    "src/pipeline",
    "src/utils",

    #Notebooks
    "NoteBooks",

    #Data
    "data/raw",
    "data/processed",
    "data/interim",
    "data/external",

    #Artifacts
    "artifacts/model",
    "artifacts/reports",
    "artifacts/logs",
    
    #config
    "config",
]

FILES=[
    #source init
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/components/model_pusher.py",

    #configuration
    "src/configuration/__init__.py",
    "src/configuration/mongo_db_connection.py",
    "src/configuration/aws_connection.py",
    
    #cloud storage
    "src/cloud_storage/__init__.py",
    "src/cloud_storage/aws_storage.py",

    #Data Access
    "src/data_access/__init__.py",
    
    #Constants
    "src/constants/__init__.py",

    #Entity
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/entity/estimator.py",
    "src/entity/s3_estimator.py",
   
   #Exception and logger
   "src/exception/__init__.py",
   "src/logger/__init__.py",

   #Pipeline
   "src/pipeline/__init__.py",
   "src/pipeline/training_pipeline.py",
   "src/pipeline/prediction_pipeline.py",

   #Utils
   "src/utils/__init__.py",
   "src/utils/main_utils.py",

   #Notebooks
   "NoteBooks/Used_Cars_Analsyis.ipynb",

   #Root files
   "app.py",
   "demo.py",
   "requirements.txt",
   "Dockerfile",
   ".dockerignore",
   "setup.py",
   "pyproject.toml",

   #config files
   "config/model.yaml",
   "config/schema.yaml",
]

# Utility Functions

def create_directories(dir_list):
    print("\n Creating Directories \n")
    for dir_path in dir_list:
        try:
            Path(dir_path).mkdir(parents=True,exist_ok=True)
            print(f"Directory ready:{dir_path}")
        except Exception as e:
            print(f"Failed to create a directory: {dir_path}|Error:{e}")
def create_files(file_list):
    print("\n Creating Files \n")
    for file_path in file_list:
        try:
            path=Path(file_path)
            path.parent.mkdir(parents=True,exist_ok=True)

            if not path.exists():
                path.touch()
                print(f'Created: {file_path}')
            else:
                print(f"Already Existed:{file_path}")
        except Exception as e:
            print(f"Failed to create files: {file_path}|Error:{e}")
            

def main():
    print("Project setup Started")
    create_directories(directories)
    create_files(FILES)
    print("Project Structure Ready!!")

if __name__=="__main__":
    main()
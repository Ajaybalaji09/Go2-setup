from roboflow import Roboflow

# You can find your API Key in your Roboflow Dashboard settings
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")

# Access the project from the URL you provided
project = rf.workspace("khangnguyen-thqz2").project("pickleball-zwqih")
model = project.version(1).model # Change '1' to the version number you want

# Download in ONNX format
model.download("onnx")

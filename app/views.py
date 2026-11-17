from django.shortcuts import render,redirect,reverse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from app.auth import authentication
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from .models import *
from django.http import JsonResponse
import os
from .image_process import *
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.http import HttpResponse, FileResponse
import mimetypes
import zipfile
import json
from ultralytics import YOLO

# Create your views here.


def index(request):
    return render(request, 'index.html')

def register(request):
    if request.method == "POST":
        first_name = request.POST['fname']
        last_name = request.POST['lname']
        username = request.POST['email']
        password = request.POST['password']
        repassword = request.POST['repassword']
        # print(first_name, contact_no, ussername)
        verify = authentication(first_name, last_name, password, repassword )
        if verify == "success":
            user = User.objects.create_user(username = username, password = password)          #create_user
            user.email = username
            user.first_name = first_name
            user.last_name = last_name
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("log_in")
            
        else:
            messages.error(request, verify)
            return redirect("register")
            # return HttpResponse("This is Home page")
    return render(request, "register.html")

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        u_name = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = u_name, password = password)
        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("projects")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    return render(request, "log_in.html")


login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname' : request.user.first_name
    }
    return render(request, 'dashboard.html', context)



@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")


def get_image_paths(folder_path):
    """
    Get image paths within a folder.
    """
    image_paths = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            image_paths.append(file_path)
    return image_paths


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def cards(request, model_name):
    context = {
        'fname': request.user.first_name
    }

    # Path to the objects folder within the model folder
    objects_folder = os.path.join(os.getcwd(), 'media', model_name, 'Objects')

    # Check if the objects folder exists
    if os.path.exists(objects_folder):
        subfolders = os.listdir(objects_folder)
        images_dict = {}

        # Iterate over subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(objects_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # Get image paths within the subfolder
                image_paths = get_image_paths(subfolder_path)
                # Store relative image paths in the dictionary
                images_dict[subfolder] = [os.path.relpath(image_path, 'media') for image_path in image_paths]

        # Pass the dictionary of image paths to the context
        context['images_dict'] = images_dict
    else:
        # If the objects folder does not exist, return an empty dictionary
        context['images_dict'] = {}

    if 'capture_image' in request.POST:
        class_name = request.POST.get('class_name')
        coco_class = request.POST.get('coco_classes')
        
        num_images = 100
        capture_images(model_name, class_name, num_images, coco_class)
        prepare_yolo_dataset(model_name)
        messages.success(request, "Class Added Successfully!!!")
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'train' in request.POST:
        train_yolo_model(model_name)
        messages.success(request, "Your Model is Trained Successfully!!")
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'test' in request.POST:
        model_path = os.path.join(os.getcwd(), 'media', model_name, 'runs', 'detect', 'train', 'weights', 'best.pt')

        # Load the YOLO model
        model = YOLO(model_path)

        # Initialize webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting live object detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO prediction
            results = model(frame)[0]

            label_to_show = ""
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0].item()
                label = f"{model.names[cls]} {conf:.2f}"
                label_to_show = model.names[cls]  # Update label to show at top-left

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw top-left label with black background
            if label_to_show:
                text_size, _ = cv2.getTextSize(label_to_show, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = text_size
                cv2.rectangle(frame, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), -1)
                cv2.putText(frame, label_to_show, (10, 10 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display result
            cv2.imshow("YOLOv8 Live Detection", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'download_model' in request.POST:
        # Paths to the files
        model_file_path = os.path.join(os.getcwd(), 'media', model_name, 'runs', 'detect', 'train', 'weights', 'best.pt')
        code_file_path = os.path.join(os.getcwd(), 'codes', 'image_classification.py')

        # Prepare the response
        response = HttpResponse(content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{0}.zip"'.format(model_name)

        # Create a zip file
        with zipfile.ZipFile(response, 'w') as zip_file:
            # Add the model file
            zip_file.write(model_file_path, arcname=os.path.basename(model_file_path))
            # Add the code file
            zip_file.write(code_file_path, arcname=os.path.basename(code_file_path))

        return response
    return render(request, 'cards.html', context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def projects(request):
    context = {
        'fname': request.user.first_name
    }
    if request.method == "POST":
        if "image_train" in request.POST:
            print("Hello 1")
            project_name = request.POST['project_name']  # Get the project name from the request GET parameters
            model_name = request.POST['model_name']  # Get the model name from the request GET parameters
            
            if model_name:  # Check if model_name is not empty
                media_folder = os.path.join(os.getcwd(), 'media')  # Get the path to the media folder
                model_folder = os.path.join(media_folder, model_name)  # Create the path to the model folder
                
                # Check if the model folder already exists
                if os.path.exists(model_folder):
                    messages.info(request, "Project Already Exists")
                    return redirect("projects")  # Return a response indicating that the project already exists
                
                # If the model folder doesn't exist, create it
                os.makedirs(model_folder)
                messages.success(request, "Project Created Successfully")
                return redirect(reverse("cards", kwargs={'model_name': model_name}))  # Return a success response

    return render(request, 'projects.html', context)



def train_model(request):
    if request.method == 'POST':
        data = request.POST.dict()
        print(data)
        # model_name = data['model']
        # classes = data['classes']

        # # Create directory for the model
        # model_directory = os.path.join(settings.MEDIA_ROOT, model_name)
        # os.makedirs(model_directory, exist_ok=True)

        # # Create subdirectories for each class and save images
        # for class_data in classes:
        #     class_name = class_data['name']
        #     class_directory = os.path.join(model_directory, class_name)
        #     os.makedirs(class_directory, exist_ok=True)

        #     # Save images to class directory
        #     for index, image_url in enumerate(class_data['images']):
        #         image_content = requests.get(image_url).content  # If image URL is provided
        #         with open(os.path.join(class_directory, f'image_{index}.jpg'), 'wb') as f:
        #             f.write(image_content)

        return JsonResponse({'message': 'Model trained successfully'}, status=200)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
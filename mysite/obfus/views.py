from .models import Participant
from django.views.generic import CreateView
from .forms import PostForm
from django.shortcuts import redirect
from django.shortcuts import render
from PIL import Image
from django.conf import settings
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)
# Create your views here.
class CreatePostView(CreateView):
    model = Participant
    form_class = PostForm
    template_name = 'obfus/post.html'
    def form_valid(self, form):
        form.save()
        part_num = form.cleaned_data.get('participant_num')
        return redirect('obfus:post_detail', part_num)

def viewPicDeepfake(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.deepfake_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicBlurred(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.blurred_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicPixelated(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.pixelated_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicMasked(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.masked_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicAsIs(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.participant_pic.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def postDetail(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    image = Image.open(participant.participant_pic)
    path_to_cropped = settings.MEDIA_ROOT + "\\images\\cropped\\"
    width, height = image.size
    filename = participant.participant_pic.url[participant.participant_pic.url.rfind("/") + 1:]
    left_image = image.crop(((0,0,width/3,height)))
    right_image = image.crop(((width/3,0, width,height)))
    left_url=path_to_cropped + filename.replace(".", "_left.")
    right_url=path_to_cropped + filename.replace(".", "_right.")
    left_image.save(left_url)
    right_image.save(right_url)
    participant.participant_pic_left= left_url
    participant.participant_pic_right= right_url
    participant.save()

    if not participant.deepfake_obfus:
        print("Running deepfake")
        get_deepfake(left_url, participant, filename, right_image)

    if not participant.blurred_obfus:
        print("Running blurred")
        get_blurred(participant, left_url, filename, right_image)

    if not participant.pixelated_obfus:
        print("Running pixelated")
        get_pixelated(participant, left_url, filename, right_image)

    if not participant.masked_obfus:
        print("Running masked")
        get_masked(participant, left_url, filename, right_image)

    context = {'participant': participant}
    return render(request, 'obfus/index.html', context)

def get_deepfake(left_url, participant, filename, right_image):
    path_to_deepfake = settings.MEDIA_ROOT + "\\images\\deepfake\\"
    path_deep = os.path.join(path_to_deepfake, filename.replace(".", "_deepfake_left."))
    path_deep_final = os.path.join(path_to_deepfake, filename.replace(".", "_deepfake."))
    command = "C:/Users/dom/Desktop/deepFakePrivacyProtector/mysite/DeepPrivacy/deep.bat " \
              + left_url + " " + path_deep
    os.system(command)
    image_left_deepfake = Image.open(path_deep)
    image_deepfake = get_concat_h(image_left_deepfake, right_image)
    image_deepfake.save(path_deep_final)
    participant.deepfake_obfus=path_deep_final
    participant.save()

def get_blurred(participant, left_url, filename, right_image):
    path_to_blurred = settings.MEDIA_ROOT + "\\images\\blurred\\"
    path_blurred_left = os.path.join(path_to_blurred, filename.replace(".", "_blurred_left."))
    path_blurred_final = os.path.join(path_to_blurred, filename.replace(".", "_blurred."))

    image = cv2.imread(left_url)
    result_image = image.copy()

    face_cascade_name = "C:\\Users\\dom\\Desktop\\deepFakePrivacyProtector\\mysite\\obfus\\haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier()

    face_cascade.load(face_cascade_name)

    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.equalizeHist(grayimg)

    faces = face_cascade.detectMultiScale(
        grayimg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Faces detected")

    if len(faces) != 0:  # If there are faces in the images
        for f in faces:  # For each face in the image

            # Get the origin co-ordinates and the length and width till where the face extends
            x, y, w, h = [v for v in f]

            # get the rectangle img around all the faces
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 5)
            sub_face = image[y:y + h, x:x + w]
            # apply a gaussian blur on this new recangle image
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            # merge this blurry rectangle to our final image
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    cv2.imwrite(path_blurred_left, result_image)
    image_left = Image.open(path_blurred_left)
    image_blurred = get_concat_h(image_left, right_image)
    image_blurred.save(path_blurred_final)
    participant.blurred_obfus=path_blurred_final
    participant.save()

def get_pixelated(participant, left_url, filename, right_image):
    path_to_pixelated = settings.MEDIA_ROOT + "\\images\\pixelated\\"
    path_pixelated_left = os.path.join(path_to_pixelated, filename.replace(".", "_pixelated_left."))
    path_pixelated_final = os.path.join(path_to_pixelated, filename.replace(".", "_pixelated."))

    image = cv2.imread(left_url)
    result_image = image.copy()

    face_cascade_name = "C:\\Users\\dom\\Desktop\\deepFakePrivacyProtector\\mysite\\obfus\\haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier()

    face_cascade.load(face_cascade_name)

    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.equalizeHist(grayimg)

    width, height, _ = image.shape

    wa,ha = (8,8)

    faces = face_cascade.detectMultiScale(
        grayimg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Faces detected")

    if len(faces) != 0:  # If there are faces in the images
        for f in faces:  # For each face in the image

            # Get the origin co-ordinates and the length and width till where the face extends
            x, y, w, h = [v for v in f]

            # get the rectangle img around all the faces
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 5)
            sub_face = image[y:y + h, x:x + w]

            sub_face = cv2.resize(sub_face, (wa,ha), interpolation=cv2.INTER_LINEAR)
            sub_face = cv2.resize(sub_face,(w,h), interpolation=cv2.INTER_NEAREST)
            # merge this blurry rectangle to our final image
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    cv2.imwrite(path_pixelated_left, result_image)
    image_left = Image.open(path_pixelated_left)
    image_blurred = get_concat_h(image_left, right_image)
    image_blurred.save(path_pixelated_final)
    participant.pixelated_obfus=path_pixelated_final
    participant.save()

def get_masked(participant, left_url, filename, right_image):
    path_to_masked = settings.MEDIA_ROOT + "\\images\\masked\\"
    path_masked_left = os.path.join(path_to_masked, filename.replace(".", "_masked_left."))
    path_masked_final = os.path.join(path_to_masked, filename.replace(".", "_masked."))

    image = cv2.imread(left_url)
    result_image = image.copy()

    face_cascade_name = "C:\\Users\\dom\\Desktop\\deepFakePrivacyProtector\\mysite\\obfus\\haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier()

    face_cascade.load(face_cascade_name)

    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.equalizeHist(grayimg)

    faces = face_cascade.detectMultiScale(
        grayimg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Faces detected")

    if len(faces) != 0:  # If there are faces in the images
        for f in faces:  # For each face in the image

            x, y, w, h = [v for v in f]

            cv2.rectangle(result_image, (x,y),(x+w,y+h),(128, 128, 128), -1)


    cv2.imwrite(path_masked_left, result_image)
    image_left = Image.open(path_masked_left)
    image_masked = get_concat_h(image_left, right_image)
    image_masked.save(path_masked_final)
    participant.masked_obfus = path_masked_final
    participant.save()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
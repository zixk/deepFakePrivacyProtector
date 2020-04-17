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
from django.contrib.staticfiles.storage import staticfiles_storage
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

def viewPicAvatar(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.avatar_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicDeepfakeBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.bystander_deepfake_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)


def viewPicBlurredBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.bystander_blurred_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)


def viewPicPixelatedBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.bystander_pixelated_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)


def viewPicMaskedBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.bystander_masked_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)


def viewPicAsIsBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.not_participant_pic.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)

def viewPicAvatarBystander(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    context = {'participant_pic': participant.bystander_avatar_obfus.url, "participant_num": participant.participant_num}
    return render(request, 'obfus/detail.html', context)


def postDetail(request, participant_num):
    participant = Participant.objects.get(participant_num=participant_num)
    image = Image.open(participant.participant_pic)
    bystander_image = Image.open(participant.not_participant_pic)
    path_to_cropped = settings.MEDIA_ROOT + "\\images\\cropped\\"
    width, height = image.size
    bystander_width, bystander_height = bystander_image.size
    filename = participant.participant_pic.url[participant.participant_pic.url.rfind("/") + 1:]
    bystander_filename = participant.not_participant_pic.url[participant.not_participant_pic.url.rfind("/") + 1:]
    left_image = image.crop(((0, 0, width / 2, height)))
    right_image = image.crop(((width / 2, 0, width, height)))
    bystander_left_image = bystander_image.crop(((0, 0, bystander_width / 2, bystander_height)))
    bystander_right_image = bystander_image.crop(((bystander_width / 2, 0, bystander_width, bystander_height)))
    left_url = path_to_cropped + filename.replace(".", "_left.")
    right_url = path_to_cropped + filename.replace(".", "_right.")
    bystander_left_url = path_to_cropped + bystander_filename.replace(".", "_left.")
    bystander_right_url = path_to_cropped + bystander_filename.replace(".", "_right.")
    left_image.save(left_url)
    right_image.save(right_url)
    bystander_left_image.save(bystander_left_url)
    bystander_right_image.save(bystander_right_url)
    participant.participant_pic_left = left_url
    participant.participant_pic_right = right_url
    participant.not_participant_pic_left = bystander_left_url
    participant.not_participant_pic_right = bystander_right_url
    participant.save()

    if not participant.avatar_obfus:
        print("Running avatar")
        path_final = get_avatar(left_url, participant, filename, right_image)
        participant.avatar_obfus = path_final

    if not participant.bystander_avatar_obfus:
        print("Running bystander avatar")
        path_final = get_avatar(bystander_left_url, participant, bystander_filename, bystander_right_image)
        participant.bystander_avatar_obfus = path_final

    if not participant.deepfake_obfus:
        print("Running deepfake")
        path_final = get_deepfake(left_url, participant, filename, right_image)
        participant.deepfake_obfus = path_final

    if not participant.bystander_deepfake_obfus:
        print("Running bystander deepfake")
        path_final = get_deepfake(bystander_left_url, participant, bystander_filename, bystander_right_image)
        participant.bystander_deepfake_obfus = path_final

    if not participant.blurred_obfus:
        print("Running blurred")
        path_final = get_blurred(participant, left_url, filename, right_image)
        participant.blurred_obfus = path_final

    if not participant.bystander_blurred_obfus:
        print("Running bystander blurred")
        path_final = get_blurred(participant, bystander_left_url, bystander_filename, bystander_right_image)
        participant.bystander_blurred_obfus = path_final

    if not participant.pixelated_obfus:
        print("Running pixelated")
        path_final = get_pixelated(participant, left_url, filename, right_image)
        participant.pixelated_obfus = path_final

    if not participant.bystander_pixelated_obfus:
        print("Running bystander pixelated")
        path_final = get_pixelated(participant, bystander_left_url, bystander_filename, bystander_right_image)
        participant.bystander_pixelated_obfus = path_final

    if not participant.masked_obfus:
        print("Running masked")
        path_final = get_masked(participant, left_url, filename, right_image)
        participant.masked_obfus = path_final

    if not participant.bystander_masked_obfus:
        print("Running bystander masked")
        path_final = get_masked(participant, bystander_left_url, bystander_filename, bystander_right_image)
        participant.bystander_masked_obfus = path_final

    participant.save()

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
    return path_deep_final



def get_avatar(left_url, participant, filename, right_image):
    path_to_avatar = settings.MEDIA_ROOT + "\\images\\avatar\\"
    path_avatar_left = os.path.join(path_to_avatar, filename.replace(".", "_avatar_left."))
    path_avatar_final = os.path.join(path_to_avatar, filename.replace(".", "_avatar."))

    if participant.gender == "Male" and participant.skin_tone == "Light":
        pic_url = staticfiles_storage.path('images/lightmale.png')
        avatar_image = cv2.imread(pic_url, cv2.IMREAD_UNCHANGED)
    if participant.gender == "Male" and participant.skin_tone == "Dark":
        pic_url = staticfiles_storage.path('images/darkmale.png')
        avatar_image = cv2.imread(pic_url, cv2.IMREAD_UNCHANGED)
    if participant.gender == "Female" and participant.skin_tone == "Light":
        pic_url = staticfiles_storage.path('images/lightfemale.png')
        avatar_image = cv2.imread(pic_url, cv2.IMREAD_UNCHANGED)
    if participant.gender == "Female" and participant.skin_tone == "Dark":
        pic_url = staticfiles_storage.path('images/darkfemale.png')
        avatar_image = cv2.imread(pic_url, cv2.IMREAD_UNCHANGED)

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
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Faces detected")

    if len(faces) != 0:  # If there are faces in the images
        for f in faces:  # For each face in the image

            x, y, w, h = [v for v in f]
            face_area = image[y-20:y+h+20,x-20:x+w+20]
            face_avatar = cv2.resize(avatar_image, (w+40,h+40), interpolation=cv2.INTER_CUBIC)
            result = transparentOverlay(face_area, face_avatar)
            result_image[y-20:y+h+20,x-20:x+w+20] = result

    cv2.imwrite(path_avatar_left, result_image)
    image_left = Image.open(path_avatar_left)
    image_avatar = get_concat_h(image_left, right_image)
    image_avatar.save(path_avatar_final)
    return path_avatar_final


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

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
        minSize=(200, 200),
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
            # apply a gaussian blur on this new recangle image
            sub_face = cv2.GaussianBlur(sub_face, (51, 51), 30)
            # merge this blurry rectangle to our final image
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    cv2.imwrite(path_blurred_left, result_image)
    image_left = Image.open(path_blurred_left)
    image_blurred = get_concat_h(image_left, right_image)
    image_blurred.save(path_blurred_final)
    return path_blurred_final


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

    wa, ha = (8, 8)

    faces = face_cascade.detectMultiScale(
        grayimg,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
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

            sub_face = cv2.resize(sub_face, (wa, ha), interpolation=cv2.INTER_LINEAR)
            sub_face = cv2.resize(sub_face, (w, h), interpolation=cv2.INTER_NEAREST)
            # merge this blurry rectangle to our final image
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

    cv2.imwrite(path_pixelated_left, result_image)
    image_left = Image.open(path_pixelated_left)
    image_blurred = get_concat_h(image_left, right_image)
    image_blurred.save(path_pixelated_final)
    return path_pixelated_final


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
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print("Faces detected")

    if len(faces) != 0:  # If there are faces in the images
        for f in faces:  # For each face in the image

            x, y, w, h = [v for v in f]

            cv2.rectangle(result_image, (x, y), (x + w, y + h), (128, 128, 128), -1)

    cv2.imwrite(path_masked_left, result_image)
    image_left = Image.open(path_masked_left)
    image_masked = get_concat_h(image_left, right_image)
    image_masked.save(path_masked_final)
    return path_masked_final


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

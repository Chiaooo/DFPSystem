from PIL import Image, ImageDraw
import face_recognition
import os


def eye_cropper(folder):
    # 新图片生成计数器
    count = 0

    # 遍历图片
    for file in os.listdir(folder):

        # 使用face_recognition
        image = face_recognition.load_image_file(folder + '/' + file)

        # 创建面部特征的坐标
        face_landmarks_list = face_recognition.face_landmarks(image)

        # 眼睛坐标
        eyes = []
        try:
            eyes.append(face_landmarks_list[0]['left_eye'])
            eyes.append(face_landmarks_list[0]['right_eye'])

        # 若是video,则用break
        except:
            continue

        # 设眼睛最大时的x, y坐标
        for eye in eyes:
            x_max = max([coordinate[0] for coordinate in eye])
            x_min = min([coordinate[0] for coordinate in eye])
            y_max = max([coordinate[1] for coordinate in eye])
            y_min = min([coordinate[1] for coordinate in eye])

            # 设x, y范围
            x_range = x_max - x_min
            y_range = y_max - y_min

            # 扩大范围，新增矩形坐标，扩大原先坐标覆盖面积的50%
            if x_range > y_range:
                right = round(.5 * x_range) + x_max
                left = x_min - round(.5 * x_range)
                bottom = round(((right - left) - y_range)) / 2 + y_max
                top = y_min - round(((right - left) - y_range)) / 2
            else:
                bottom = round(.5 * y_range) + y_max
                top = y_min - round(.5 * y_range)
                right = round(((bottom - top) - x_range)) / 2 + x_max
                left = x_min - round(((bottom - top) - x_range)) / 2

            # 保存图片到变量im
            im = Image.open(folder + '/' + file)

            # 以临时变量Left top right bottom裁剪
            im = im.crop((left, top, right, bottom))

            # 调整图像大小，方便模型输入
            im = im.resize((80, 80))

            im.save('data/train/new_closed_eyes/eye_crop_closed' + str(count) + '.jpg')

            # 计数器工作
            count += 1

            # 生成进度
            if count % 200 == 0:
                print('已成功生成并导入：'+str(count))


eye_cropper('data/closed_eyes_face_data')

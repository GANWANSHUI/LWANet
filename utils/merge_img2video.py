import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import  pdb

def merge_img2video(image_path, video_path):
    # path = image_path # 图片序列所在目录，文件名：0.jpg 1.jpg ...
    # dst_path =  video_path   #r'F:\dst\result.mp4' # 生成的视频路径

    filelist = os.listdir(image_path)
    filepref = [os.path.splitext(f)[0] for f in filelist]



    filepref.sort(key = int) # 按数字文件名排序
    #filepref= sorted(filepref,key=lambda x: int(x[:-6]))  # 按数字文件名排序

    #pdb.set_trace()

    filelist = [f + '.png' for f in filepref]

    # size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
    #         int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    width = 1216
    height = 320

    # width = 1238
    # height = 374
    fps = 30

    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width , height))

    #for file in filelist[5:-6]:
    for file in filelist:
        if file.endswith('.png'):
            file = os.path.join(image_path, file)
            print("file:", file)
            img = cv2.imread(file)
            print("img:", img.shape)


            # img = img[54:,22:,:]
            # #img = img[50:, 10:, :]


            print("img:", img.shape)
            #img = np.hstack((img, img))  # 如果并排两列显示
            vw.write(img)


    vw.release()



def merge_video():

    videoLeftUp = cv2.VideoCapture('/home/wsgan/LWANet/results/video/0028/raw_img_title.mp4')
    videoLeftDown = cv2.VideoCapture('/home/wsgan/LWANet/results/video/0028/GT_supervise/GT_supervise_subtitile.mp4')
    videoRightUp = cv2.VideoCapture('/home/wsgan/LWANet/results/video/0028/self_supervise/Self_supervise_subtitle.mp4')
    videoRightDown = cv2.VideoCapture('/home/wsgan/LWANet/results/video/0028/no_supervise/No_supervise_subtitle.mp4')

    fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter('/home/wsgan/LWANet/results/video/0028/merge0028.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown, frameLeftDown = videoLeftDown.read()
    successRightUp, frameRightUp = videoRightUp.read()
    successRightDown, frameRightDown = videoRightDown.read()

    while successLeftUp and successLeftDown and successRightUp and successRightDown:
        frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightDown = cv2.resize(frameRightDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

        frameUp = np.hstack((frameLeftUp, frameRightUp))
        frameDown = np.hstack((frameLeftDown, frameRightDown))
        frame = np.vstack((frameUp, frameDown))

        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()
        successRightUp, frameRightUp = videoRightUp.read()
        successRightDown, frameRightDown = videoRightDown.read()

    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()
    videoRightUp.release()
    videoRightDown.release()



def add_subtitle(video_path, save_path):

    cap = cv2.VideoCapture(video_path)  # 读取视频

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (1216, 320))  # 输出视频参数设置

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # 在 frame 上显示一些信息
            img_PIL = Image.fromarray(frame[..., ::-1])  # 转成 array
            font = ImageFont.truetype('UbuntuMono-B.ttf', 40)  # 字体设置，Windows系统可以在 "C:\Windows\Fonts" 下查找
            text1 = "Self_supervise"

            for i, te in enumerate(text1):
                # position = (50, 10 + i * 50)
                position = (10 + i * 20, 20 )
                draw = ImageDraw.Draw(img_PIL)
                draw.text(position, te, font=font, fill=(255, 0, 0))

            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

            # write the frame
            #cv2.imshow('frame', frame)
            out.write(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    #cv2.destroyAllWindows()






if __name__ == "__main__":
    print('start')
    #merge_img2video('/home/wsgan/LWANet/results/video/0071/self_supervise/disparity', '/home/wsgan/LWANet/results/video/0071/self_supervise/Self_supervise.mp4' )
    merge_video()
    #add_subtitle('/home/wsgan/LWANet/results/video/0071/self_supervise/Self_supervise.mp4', '/home/wsgan/LWANet/results/video/0071/self_supervise/Self_supervise_subtitle.mp4')
    print('end')
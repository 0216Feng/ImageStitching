import cv2
import numpy as np
import os
import sys

class Matchers:
    def __init__(self):
        self.sift = cv2.SIFT_create()  # 修改为使用 SIFT
        FLANN_INDEX_KDTREE = 1  # 使用 FLANN_INDEX_KDTREE 对应 SIFT
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, img1, img2, direction=None):
        imageSet1 = self.getSIFTFeatures(img1)  # 更新方法名
        imageSet2 = self.getSIFTFeatures(img2)  # 更新方法名
        print("Direction:", direction)
        
        matches = self.flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (_, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, _) in good])

            H, _ = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
            return H
        return None

    def getSIFTFeatures(self, im):  # 更新方法名称
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

class Stitch:
    def __init__(self, image_folder):
        self.images = self.load_images(image_folder)
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = Matchers()
        self.prepare_lists()

    def load_images(self, folder):
        filenames = [f for f in os.listdir(folder) if f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        images = [cv2.resize(cv2.imread(os.path.join(folder, f)), (480, 320)) for f in filenames]
        return images

    def prepare_lists(self):
        print(f"Number of images: {self.count}")
        self.centerIdx = self.count // 2
        print(f"Center index image: {self.centerIdx}")
        self.center_im = self.images[self.centerIdx]
        self.left_list = self.images[:self.centerIdx + 1]
        self.right_list = self.images[self.centerIdx + 1:]
        print("Image lists prepared")

    def leftshift(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            print("Homography is:", H)
            xh = np.linalg.inv(H)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]
            f1 = np.dot(xh, np.array([0, 0, 1]))
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            print("Image dsize =>", dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp
        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            print("Homography:", H)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            tmp = self.mix_and_match(self.leftImage, tmp)
            gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                tmp = tmp[y:y+h, x:x+w]
            self.leftImage = tmp

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        
        for i in range(i1x):
            for j in range(i1y):
                if np.array_equal(leftImage[j, i], [0, 0, 0]) and np.array_equal(warpedImage[j, i], [0, 0, 0]):
                    warpedImage[j, i] = [0, 0, 0]
                elif np.array_equal(warpedImage[j, i], [0, 0, 0]):
                    warpedImage[j, i] = leftImage[j, i]
                elif not np.array_equal(leftImage[j, i], [0, 0, 0]):
                    b, g, r = warpedImage[j, i]
                    bl, gl, rl = leftImage[j, i]
                    warpedImage[j, i] = [bl, gl, rl]

        return warpedImage

    def showImage(self):
        cv2.imshow("Stitched Image", self.leftImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_folder = 'images'  # 指定图片所在的文件夹
    stitcher = Stitch(image_folder)
    stitcher.leftshift()
    stitcher.rightshift()
    stitcher.showImage()
    cv2.imwrite("stitched_output.jpg", stitcher.leftImage)
    print("Stitched image saved as stitched_output.jpg")

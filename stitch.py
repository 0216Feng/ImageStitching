import cv2
import os
import numpy as np

def cvshow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


# 全景拼接
def siftimg_rightlignment(img_right, img_left):
    """图像拼接"""
    # 使用新版SIFT
    sift = cv2.SIFT_create()
    
    # 转换为灰度图并提取特征点
    gray1 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) > 4:
        # 获取匹配点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # 计算变换矩阵
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
        
        # 获取图像尺寸
        h1, w1 = img_right.shape[:2]
        h2, w2 = img_left.shape[:2]
        
        # 计算变换后的边界
        corners = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 计算边界和偏移
        [xmin, ymin] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
        
        # 确保偏移量为正
        t = [-xmin if xmin < 0 else 0, -ymin if ymin < 0 else 0]
        
        # 计算输出图像大小
        width = xmax - xmin
        height = max(ymax - ymin, h2)
        
        # 更新变换矩阵
        Ht = np.array([[1,0,t[0]], [0,1,t[1]], [0,0,1]])
        H = Ht.dot(H)
        
        # 创建输出图像
        result = cv2.warpPerspective(img_right, H, (width, height))
        
        # 确保坐标有效
        y_offset = t[1]
        x_offset = t[0]
        result[y_offset:y_offset+h2, x_offset:x_offset+w2] = img_left
        
        # 裁剪黑边
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            result = result[y:y+h, x:x+w]
        
        return result
    return None
    
def read_images(folder):
    """读取文件夹中的所有图片"""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def preprocess_images(images):
    """调整所有图片为相同大小"""
    if not images:
        return []
    # 以第一张图片的大小为基准
    height, width = images[0].shape[:2]
    processed = []
    for img in images:
        resized = cv2.resize(img, (width, height))
        processed.append(resized)
    return processed

def stitch_images(images):
    """拼接多张图片"""
    if not images:
        return None
    
    result = images[0]
    for i in range(1, len(images)):
        result = siftimg_rightlignment(images[i], result)
        if result is None:
            print(f"拼接第 {i} 张图片失败")
            return None
    return result
      

def main():
    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 读取图片
    images = read_images('images')
    if not images:
        print("未找到任何图片")
        return
    
    # 预处理图片
    processed_images = preprocess_images(images)
    
    # 拼接图片
    result = stitch_images(processed_images)
    
    if result is not None:
        # 保存结果
        output_path = os.path.join('output', 'panorama.jpg')
        cv2.imwrite(output_path, result)
        print(f"全景图已保存至: {output_path}")
        
        # 显示结果
        cvshow('Final Panorama', result)
    else:
        print("图片拼接失败")

if __name__ == "__main__":
    main()
